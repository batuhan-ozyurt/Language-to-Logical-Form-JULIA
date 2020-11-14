training_file = "C:\\Users\\batuh\\Desktop\\COMP541\\Datasets\\seq2seq_atis\\train.txt"

inputs = []
outputs = []
input_tokens = []
output_tokens = []
size_input = 1 #max input sequence length. It will be updated later.
size_output = 1 #max output sequence length. It will be updated later.

f = open(training_file)

#reading the training file ans storing the contents in "inputs" and "outputs" variables. also, uniwuee tokens are stored
#in "input_tokens" and "output_tokens" variables.

while ! eof(f)
    s = readline(f)
    input, output = split(s, "\t")
    append!(inputs, [input])
    append!(outputs, [output])

    tokens = split(input, " ")
    if size(tokens)[1] > size_input
        size_input = size(tokens)[1]
    end
    for token in tokens
        if !(token in input_tokens)
            append!(input_tokens, [token])
        end
    end

    tokens = split(output, " ")
    if size(tokens)[1] > size_output
        size_output = size(tokens)[1]
    end

    for token in tokens
        if !(token in output_tokens)
            append!(output_tokens, [token])
        end
    end
end

K = size(input_tokens)[1]
L = size(output_tokens)[1]

#creating one-hot vectors for each token

one_hot_to_token_input = Dict()
one_hot_to_token_output = Dict()
token_to_one_hot_input = Dict()
token_to_one_hot_output = Dict()

i = 0
for token in input_tokens
    i += 1
    x = zeros(K)
    x[i] = 1
    one_hot_to_token_input[x] = token
    token_to_one_hot_input[token] = x

end

i = 0
for token in output_tokens
    i += 1
    y = zeros(L)
    y[i] = 1
    one_hot_to_token_output[y] = token
    token_to_one_hot_output[token] = y
end

#the input and output sequences are transformed into their one-hot representations:
model_input = []
model_output = []

for input in inputs
    tokens = split(input, " ")
    seq = zeros(size_input*K)
    i = 0

    for token in tokens
        seq[(K*i+1):(K*(i+1))] = token_to_one_hot_input[token]
        i += 1
    end

    append!(model_input, [seq])
end

for output in outputs
    tokens = split(output, " ")
    seq = zeros(size_output*L)
    i = 0

    for token in tokens
        seq[(L*i+1):(L*(i+1))] = token_to_one_hot_output[token]
        i += 1
    end

    append!(model_output, [seq])

end

#outputs are created randomly and stored in "predictions" variable.
predictions = []
for j in 1:size(model_input)[1]
    prediction = zeros(size_output*L)
    length = count(i->(i==1), model_input[j])
    for k in 0:length+14
        a = rand(1:L)
        prediction[k*L + a] = 1
    end
    append!(predictions, [prediction])
end

#below, F1 scores are calculated.
j = 0
F1_scores = 0.0
for prediction in predictions
    j += 1
    pred_length = count(i->(i==1), prediction)
    gold_length = count(i->(i==1), model_output[j])

    gold_tokens = split(outputs[j], " ")
    correct = 0

    pred_tokens = []

    for k in 0:pred_length-1
        vector1 = prediction[(k*L+1):((k+1)*L)]
        token1 = one_hot_to_token_output[vector1]
        append!(pred_tokens, [token1])
    end

    for pred_token in pred_tokens
        for gold_token in gold_tokens
            if (pred_token == gold_token)
                correct += 1
                deleteat!(gold_tokens, findfirst(x -> x==gold_token, gold_tokens))
                break
            end
        end
    end

    precision = correct / pred_length
    recall = correct / gold_length

    if precision == 0 || recall == 0
        F1_scores += 0
    else
        F1_scores += 2*precision*recall / (precision + recall)
    end

end

F1_score = F1_scores / size(model_input)[1]
