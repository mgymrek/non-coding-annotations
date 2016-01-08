require 'dpnn'
require 'nn'
require 'optim'

if cuda then
    require 'cunn'
    require 'cutorch'
    require 'inn'
end

require 'accuracy'
require 'batcher'

ConvNet = {}

function ConvNet:__init()
    obj = {}
    setmetatable(obj, self)
    self.__index = self
    return obj
end

----------------------------------------------------------------
-- adjust_final
--
-- Reset the final layer for new targets.
----------------------------------------------------------------
function ConvNet:adjust_final(num_targets, target_type)
    -- re-specify num targets
    self.num_targets = num_targets

    -- re-specify target_type
    local target_type_seed = self.target_type
    self.target_type = target_type or self.target_type

    -- remove nonlinearity
    if target_type_seed == "binary" or target_type_seed == "positive" then
        self.model:remove()
    end

     -- save input size
    local hidden_in = self.model.modules[#self.model].weight:size(2)

    -- remove Linear
    self.model:remove()

    -- add final linear
    self.model:add(nn.Linear(hidden_in, self.num_targets))

    if self.target_type == "binary" then
        -- add final nonlinearity
        self.model:add(nn.Sigmoid())

        -- binary cross-entropy loss
        self.criterion = nn.BCECriterion()

    elseif self.target_type == "positive" then
        -- binary cross-entropy loss
        self.model:add(nn.ReLU())

        -- mean-squared error loss
        self.criterion = nn.MSECriterion()
    else
        -- mean-squared error loss
        self.criterion = nn.MSECriterion()
    end

    self.criterion.sizeAverage = false

    -- cuda
    if cuda then
        print("Running on GPU.")
        self.model:cuda()
        self.criterion:cuda()
    end

    -- retrieve parameters and gradients
    self.parameters, self.gradParameters = self.model:getParameters()

    -- print model summary
    print(self.model)
end

----------------------------------------------------------------
-- adjust_optim
--
-- Reset the optim state and set optimization hyper-params
----------------------------------------------------------------
function ConvNet:adjust_optim(job)
    -- reset rmsprop state
    self.optim_state.m = nil

    -- number of examples per weight update
    self.batch_size = job.batch_size or 128

    -- base learning rate
    self.learning_rate = job.learning_rate or 0.002

    -- gradient update momentum
    self.momentum = job.momentum or 0.98
end


----------------------------------------------------------------
-- build
--
-- Build the network using the job parameters and data
-- attributes.
----------------------------------------------------------------
function ConvNet:build(job, init_depth, init_len, num_targets)
    -- parse network structure parameters
    self:setStructureParams(job)

    -- initialize model sequential
    self.model = nn.Sequential()

    -- store useful values
    self.num_targets = num_targets
    local depth = init_depth
    local seq_len = init_len

    -- convolution layers
    for i = 1,self.conv_layers do
        -- convolution
        if i == 1 or self.conv_conn[i-1] == 1 then
            -- fully connected convolution
            self.model:add(nn.SpatialConvolution(depth, self.conv_filters[i], self.conv_filter_sizes[i], 1))
        else
            -- randomly connected convolution
            num_to = torch.round(depth*self.conv_conn[i-1])
            conn_matrix = nn.tables.random(depth, self.conv_filters[i], num_to)
            self.model:add(nn.SpatialConvolutionMap(conn_matrix, self.conv_filter_sizes[i], 1))
        end

        -- update sequence length for filter pass
        seq_len = seq_len - self.conv_filter_sizes[i] + 1

        -- batch normalization (need to figure out how to ditch the bias above)
        if self.batch_normalize then
            self.model:add(nn.SpatialBatchNormalization(self.conv_filters[i]))
        end

        -- nonlinearity
        self.model:add(nn.ReLU())

        -- dropout
        if self.conv_dropouts[i] > 0 then
            self.model:add(nn.Dropout(self.conv_dropouts[i]))
        end

        -- pooling
        if self.pool_width[i] > 1 then
            if self.pool_op == "max" then
                -- trimming the seq
                pseq_len = math.ceil(seq_len / self.pool_width[i])
                pool_mod = nn.SpatialMaxPooling(self.pool_width[i], 1)
                pool_mod:ceil()
                self.model:add(pool_mod)
            else
                pseq_len = math.floor(seq_len / self.pool_width[i])
                self.model:add(inn.SpatialStochasticPooling(self.pool_width[i],1))
            end

            seq_len = pseq_len
        end

        -- update helper
        depth = self.conv_filters[i]
    end

    -- too much pooling
    if seq_len <= 0 then
        return false
    end

    -- prep for fully connected layers
    hidden_in = depth*seq_len
    self.model:add(nn.Reshape(hidden_in))

    -- fully connected hidden layers
    for i =1,self.hidden_layers do
        -- linear transform
        self.model:add(nn.Linear(hidden_in, self.hidden_units[i]))

        -- batch normalization (need to figure out how to ditch the bias above)
        if self.batch_normalize then
            self.model:add(nn.BatchNormalization(self.hidden_units[i]))
        end

        -- nonlinearity
        self.model:add(nn.ReLU())

        -- dropout
        if self.hidden_dropouts[i] > 0 then
            self.model:add(nn.Dropout(self.hidden_dropouts[i]))
        end

        -- update helper
        hidden_in = self.hidden_units[i]
    end

    if self.target_type == "binary" then
        -- final layer w/ target priors as initial biases
        final_linear = nn.Linear(hidden_in, self.num_targets)
        -- target_priors = targets:mean(1):squeeze()
        -- biases_init = -torch.log(torch.pow(target_priors, -1) - 1)
        -- final_linear.bias = biases_init
        self.model:add(final_linear)
        self.model:add(nn.Sigmoid())

        -- binary cross-entropy loss
        self.criterion = nn.BCECriterion()

    elseif self.target_type == "positive" then
        -- final layer
        self.model:add(nn.Linear(hidden_in, self.num_targets))
        self.model:add(nn.ReLU())

        -- mean-squared error loss
        self.criterion = nn.MSECriterion()

    else
        -- final layer
        self.model:add(nn.Linear(hidden_in, self.num_targets))

        -- mean-squared error loss
        self.criterion = nn.MSECriterion()
    end

    self.criterion.sizeAverage = false

    -- cuda
    if cuda then
        print("Running on GPU.")
        self.model:cuda()
        self.criterion:cuda()
    end

    -- retrieve parameters and gradients
    self.parameters, self.gradParameters = self.model:getParameters()

    -- print model summary
    print(self.model)

    -- the following code breaks the program, but
    -- it's interesting to see those counts!

    -- print(string.format("Sum:      %7d parameters",(#self.parameters)[1]))
    -- for i = 1,(#self.model) do
    -- 	local layer_params = self.model.modules[i]:getParameters()
    -- 	local np = 0
    -- 	if layer_params:nDimension() > 0 then
    -- 		np = (#layer_params)[1]
    -- 	end
    -- 	print(string.format("Layer %2d: %7d", i, np))
    -- end

    return true
end


----------------------------------------------------------------
-- decuda
--
-- Move the model back to the CPU.
----------------------------------------------------------------
function ConvNet:decuda()
    self.optim_state.m = self.optim_state.m:double()
    self.optim_state.tmp = self.optim_state.tmp:double()
    self.criterion:double()
    self.model:double()
    self.parameters, self.gradParameters = self.model:getParameters()
    -- self.parameters = self.parameters:double()
    -- self.gradParameters = self.gradParameters:double()
    cuda = false
end


----------------------------------------------------------------
-- get_nonlinearity
--
-- Return the module representing nonlinearity x.
----------------------------------------------------------------
function ConvNet:get_nonlinearity(x)
    nl_modules = self.model:findModules('nn.ReLU')
    return nl_modules[x]
end


----------------------------------------------------------------
-- get_final
--
-- Return the module representing the final layer.
----------------------------------------------------------------
function ConvNet:get_final()
    local layers = #self.model
    return self.model.modules[layers-1]
end


function ConvNet:load(cnn)
    for k, v in pairs(cnn) do
        self[k] = v
    end
end


----------------------------------------------------------------
-- predict
--
-- Predict targets for a new set of sequences.
-- Outputs to out_file
----------------------------------------------------------------
function ConvNet:predict(Xf, batch_size, out_file)
    local bs = batch_size or self.batch_size
    local batcher = Batcher:__init(Xf, nil, bs)

    -- open output file
    local predict_out = io.open(out_file, 'w')

    -- get first batch
    local Xb = batcher:next()

    -- while batches remain
    while Xb ~= nil do
        -- cuda
        if cuda then
            Xb = Xb:cuda()
        end

        -- predict and normalize
        local preds_batch = self.model:forward(Xb)

	-- write to output
	for si=1,(#preds_batch)[1] do
	    predict_out:write(preds_batch[{si,1}])
	    for ti=2,(#preds_batch)[2] do
	    	predict_out:write(string.format("\t%s",preds_batch[{si, ti}]))
	    end
	    predict_out:write("\n")
	end

        -- next batch
        Xb = batcher:next()

        collectgarbage()
    end
    predict_out.close()
end


----------------------------------------------------------------
-- sanitize
--
-- Clear the intermediate states in the model before
-- saving to disk.
----------------------------------------------------------------
function ConvNet:sanitize()
    local module_list = self.model:listModules()
    for _,val in ipairs(module_list) do
        for name,field in pairs(val) do
            if torch.type(field) == 'cdata' then val[name] = nil end
            if name == 'homeGradBuffers' then val[name] = nil end
            if name == 'input_gpu' then val['input_gpu'] = {} end
            if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
            if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
            if name == 'output' or name == 'gradInput' then
                val[name] = field.new()
            end
            -- batch normalization
            if name == 'buffer' or name == 'normalized' or name == 'centered' then
                val[name] = field.new()
            end
            -- max pooling
            if name == 'indices' then
                val[name] = field.new()
            end
        end
    end
end

----------------------------------------------------------------
-- sanitize
--
-- Clear the intermediate states in the model before
-- saving to disk.
----------------------------------------------------------------
function ConvNet:setStructureParams(job)
    ---------------------------------------------
    -- training
    ---------------------------------------------
    -- number of examples per weight update
    self.batch_size = job.batch_size or 128

    -- base learning rate
    self.learning_rate = job.learning_rate or 0.002

    -- gradient update momentum
    self.momentum = job.momentum or 0.98

    -- batch normaliztion
    if job.batch_normalize == nil then
        self.batch_normalize = true
    else
        self.batch_normalize = job.batch_normalize
    end

    -- leaky ReLU leak parameter
    -- self.leak = job.leak or 0.01

    -- self.conv_leak = job.conv_leak or self.leak
    -- self.final_leak = job.final_leak or self.leak

    -- normalize weight vectors to this max
    self.weight_norm = job.weight_norm or 10

    ---------------------------------------------
    -- network structure
    ---------------------------------------------
    -- number of filters per layer
    self.conv_filters = job.conv_filters or {10}
    if type(self.conv_filters) == "number" then
        self.conv_filters = {self.conv_filters}
    end

    -- or determine via scaling
    if job.conv_layers and job.conv_filters_scale then
        self.conv_layers = job.conv_layers
        for i=2,self.conv_layers do
            self.conv_filters[i] = math.ceil(job.conv_filters_scale * self.conv_filters[i-1])
        end
    end

    -- determine number of convolution layers
    self.conv_layers = #self.conv_filters

    -- convolution filter sizes
    if job.conv_filter1_size == nil then
        self.conv_filter_sizes = job.conv_filter_sizes or {10}
        if type(self.conv_filter_sizes) == "number" then
            self.conv_filter_sizes = {self.conv_filter_sizes}
        end
    else
        self.conv_filter_sizes = {job.conv_filter1_size}
        local l = 2
        while job[string.format("conv_filter%d_size",l)] ~= nil do
            self.conv_filter_sizes[l] = job[string.format("conv_filter%d_size",l)]
            l = l + 1
        end
    end

    -- or determine via scaling
    if job.conv_layers and job.conv_size_scale then
        for i=2,job.conv_layers do
            self.conv_filter_sizes[i] = math.ceil(job.conv_size_scale * self.conv_filter_sizes[i-1])
        end
    end

    -- pooling widths
    self.pool_width = table_ext(job.pool_width, 1, self.conv_layers)

    -- or determine via scaling
    if job.conv_layers and job.pool_width_scale then
        for i=2,job.conv_layers do
            self.pool_width[i] = math.ceil(job.pool_width_scale * self.pool_width[i-1])
        end
    end

    -- pooling operation ("max", "stochastic")
    self.pool_op = job.pool_op or "max"

    -- random connections (need to test this with one layer, where it becomes irrelevant)
    self.conv_conn = table_ext(job.conv_conn, 1, self.conv_layers-1)


    -- number of hidden units in the final fully connected layers
    self.hidden_units = table_ext(job.hidden_units, 500, 1)

    -- number of fully connected final layers
    self.hidden_layers = #self.hidden_units


    -- target value type ("binary", "continuous", "positive")
    self.target_type = job.target_type or "binary"

    ---------------------------------------------
    -- regularization
    ---------------------------------------------
    -- input dropout probability
    -- self.input_dropout = job.input_dropout or 0

    -- convolution dropout probabilities
    self.conv_dropouts = table_ext(job.conv_dropouts, 0, self.conv_layers)

    -- convolution gaussian noise stdev
    self.conv_gauss = table_ext(job.conv_gauss, 0, self.conv_layers)

    -- final dropout probabilities
    self.hidden_dropouts = table_ext(job.hidden_dropouts, 0, self.hidden_layers)

    -- final gaussian noise stdev
    self.hidden_gauss = table_ext(job.hidden_gauss, 0, self.hidden_layers)

    -- L2 parameter norm
    self.coef_l2 = job.coef_l2 or 0

    -- L1 parameter norm
    self.coef_l1 = job.coef_l1 or 0

end


---------------------------------------------------------------
-- train_epoch
--
-- Train the model for one epoch through the data specified by
-- batcher.
----------------------------------------------------------------
function ConvNet:train_epoch(batcher)
    local total_loss = 0

    -- collect garbage occasionaly
    local cgi = 0

    -- get first batch
    local inputs, targets = batcher:next()

    -- while batches remain
    while inputs ~= nil do
        -- cuda
        if cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= self.parameters then
                self.parameters:copy(x)
            end

            -- reset gradients
            self.gradParameters:zero()

            -- evaluate function for mini batch
            local outputs = self.model:forward(inputs)
            local f = self.criterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = self.criterion:backward(outputs, targets)
            self.model:backward(inputs, df_do)

            -- penalties
            if self.coef_l2 > 0 then
                -- add to loss
                f = f + self.coef_l2 * torch.norm(self.parameters,2)^2/2

                -- add to gradient
                self.gradParameters:add(self.coef_l2, self.parameters)
            end
            if self.coef_l1 > 0 then
                -- add to loss
                f = f + self.coef_l1 * torch.norm(self.parameters,1)

                -- add to gradient
                self.gradParameters:add(torch.sign(self.parameters):mul(self.coef_l1))
            end

            -- return f and df/dX
            return f, self.gradParameters
        end

        -- perform RMSprop step
        self.optim_state = self.optim_state or {
            learningRate = self.learning_rate,
            alpha = self.momentum
        }
        optim.rmsprop(feval, self.parameters, self.optim_state)

        -- cap weight paramaters
        self.model:maxParamNorm(self.weight_norm)

        -- accumulate loss
        total_loss = total_loss + self.criterion.output

        -- next batch
        inputs, targets = batcher:next()

        -- collect garbage occasionaly
        cgi = cgi + 1
        if cgi % 100 == 0 then
            collectgarbage()
        end
    end

    -- reset batcher
    batcher:reset()

    -- mean loss over examples
    avg_loss = total_loss / batcher.num_seqs
    return avg_loss
end


----------------------------------------------------------------
-- test
--
-- Predict targets for X and compare to Y.
----------------------------------------------------------------
function ConvNet:test(Xf, Yf, batch_size)
    -- track the loss across batches
    local loss = 0

    -- collect garbage occasionaly
    local cgi = 0

    -- create a batcher to help
    local batch_size = batch_size or self.batch_size
    local batcher = Batcher:__init(Xf, Yf, batch_size)

    -- track predictions across batches
    local preds = torch.Tensor(batcher.num_seqs, self.num_targets)
    local pi = 1

    -- get first batch
    local inputs, targets = batcher:next()

    -- while batches remain
    while inputs ~= nil do
        -- cuda
        if cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- predict
        local preds_batch = self.model:forward(inputs)

        -- accumulate loss
        loss = loss + self.criterion:forward(preds_batch, targets)

        -- copy into larger Tensor
        for i = 1,(#preds_batch)[1] do
            preds[{pi,{}}] = preds_batch[{i,{}}]:float()
            pi = pi + 1
        end

        -- next batch
        inputs, targets = batcher:next()

        -- collect garbage occasionaly
        cgi = cgi + 1
        if cgi % 100 == 0 then
            collectgarbage()
        end
    end

    -- mean loss over examples
    local avg_loss = loss / batcher.num_seqs

    -- save pred means and stds
    self.pred_means = preds:mean(1):squeeze()
    self.pred_stds = preds:std(1):squeeze()

    local Ydim = batcher.num_targets
    if self.target_type == "binary" then
        -- compute AUC
        local AUCs = torch.Tensor(Ydim)
        local roc_points = {}
        for yi = 1,Ydim do
            -- read Yi from file
            local Yi = Yf:partial({1,batcher.num_seqs},{yi,yi}):squeeze()

            -- compute ROC points
            roc_points[yi] = ROC.points(preds[{{},yi}], Yi)

            -- compute AUCs
            AUCs[yi] = ROC.area(roc_points[yi])

            collectgarbage()
        end

        return avg_loss, AUCs, roc_points
    else
        -- compute R2
        local R2s = variance_explained(Yf, preds)

        return avg_loss, R2s
    end
end


----------------------------------------------------------------
-- table_ext
--
-- Extend the table to be the given size, adding in the default
-- value to fill it.
----------------------------------------------------------------
function table_ext(try, default, size)
    -- set var to try if available or default otherwise
    var = try or default

    -- if it was a number
    if type(var) == "number" then
        -- change default to the number
        default = var

        -- make it a table
        var = {var}
    end

    -- extend the table if too small
    for i = 2,size do
        if i > #var then
            var[i] = default
        end
    end

    return var
end