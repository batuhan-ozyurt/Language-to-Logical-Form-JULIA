using Distributions
import .Iterators: cycle, Cycle, take
using IterTools
import CUDA
using CUDA

function tok_int(src_vocab_file, tgt_vocab_file)
    
    #global int2tok_input, int2tok_output
    f_src = open(src_vocab_file)
    f_tgt = open(tgt_vocab_file)
    
    tok2int_input = Dict{String,Int}() #keys: unique input tokens. values: id number of the token.
    int2tok_input = Vector{String}() #indices: numbers. values: tokens corresponding to those numbers.
    
    push!(int2tok_input, "<s>") #start token
    push!(int2tok_input, "</s>") #stop token
    push!(int2tok_input, "<n>") #nonterminal token
    #push!(int2tok_input, "(") 
    #push!(int2tok_input, ")") 
    push!(int2tok_input, "UNK") #unknown token
    
    tok2int_input["<s>"] = 1
    tok2int_input["</s>"] = 2
    tok2int_input["<n>"] = 3
    #tok2int_input["("] = 4
    #tok2int_input[")"] = 5
    tok2int_input["UNK"] = 4
    
    tok2int_output = Dict{String,Int}() #keys: unique output tokens. values: id number of the token.
    int2tok_output = Vector{String}() #keys: numbers. values: tokens corresponding to those numbers.
    
    push!(int2tok_output, "<s>") #start token
    push!(int2tok_output, "</s>") #stop token
    push!(int2tok_output, "<n>")
    #push!(int2tok_output, "(")
    #push!(int2tok_output, ")")
    push!(int2tok_output, "UNK") #unknown token
    
    tok2int_output["<s>"] = 1
    tok2int_output["</s>"] = 2
    tok2int_output["<n>"] = 3
    #tok2int_output["("] = 4
    #tok2int_output[")"] = 5
    tok2int_output["UNK"] = 4
    
    while ! eof(f_src)
        seq = readline(f_src)
        seq = chomp(seq)
        token, frequency = split(seq, "\t")
        if !haskey(tok2int_input, token) && parse(Int64, frequency) > 1
            push!(int2tok_input, token)
            tok2int_input[token] = length(int2tok_input)
        end               
    end
    
    while ! eof(f_tgt)
        seq = readline(f_tgt)
        seq = chomp(seq)
        token, frequency = split(seq, "\t")
        if !haskey(tok2int_output, token)
            push!(int2tok_output, token)
            tok2int_output[token] = length(int2tok_output)
        end               
    end
    
    Vq = length(int2tok_input) #number of unique input tokens
    Va = length(int2tok_output) #number of unique output tokens 
    
    return int2tok_input, tok2int_input, int2tok_output, tok2int_output, Vq, Va
end

function data_reader(training_file, tok2int_input, tok2int_output)
    
    data = []
    f = open(training_file)
    n_in = 0 #just to check the number of unknown tokens
    n_out = 0 #just to check the number of unknown tokens
    while ! eof(f)
        seq = readline(f)
        seq = chomp(seq)
        input, output = split(seq, "\t")
        tokens = split(input, " ")
        src = Vector{Int}() #vector that stores the token ids.        
        for token in tokens            
            if haskey(tok2int_input, token)
                push!(src, tok2int_input[token])
            else 
                push!(src, tok2int_input["UNK"])
                n_in += 1
            end 
        end
        tokens = split(output, " ")
        tgt = Vector{Int}() #vector that stores the token ids.        
        for token in tokens
            println
            if haskey(tok2int_output, token)
                push!(tgt, tok2int_output[token])
            else
                push!(tgt, tok2int_output["UNK"])
                n_out += 1
            end
        end 
        t = convert_to_tree(tgt, 1, length(tgt))
        push!(data, (src, tgt, t))
    end    
    return data, n_in, n_out
end

function list_to_string(token_ids, int2tok)
    out_str = ""
    for id in token_ids
        out_str *= int2tok[id] * " "
    end
    return rstrip(out_str)
end

function minibatch(data, batchsize)
    
    n = length(data)
    p = 0
    batch_data = Any[]
    
    if length(data) % batchsize != 0
        n = length(data)
        for i in 0:((length(data)%batchsize) - 1)
            insert!(data, n-i, data[n-i])
        end
    end
    
    while p + batchsize <= n
        
        max_seq_len = length(data[p+batchsize][1])
        enc_seq = zeros(Int64, batchsize, max_seq_len+2)
        enc_seq[:, 1] .= 1
        enc_len = Any[]
        
        for i in 1:batchsize
            seq = data[p+i][1]
            seq_len = length(seq)
            
            for j in 1:seq_len
                enc_seq[i, j+1] = seq[seq_len-j+1]
            end
            
            for k in seq_len+2:max_seq_len+2 #pad the ending with end of seq tokens
                enc_seq[i,k] = 2
            end
            
            push!(enc_len, seq_len+2)
            
        end
                
        dec_seq = Any[]
        for i in 1:batchsize
            push!(dec_seq, data[p+i][2])
        end        
        
        tree_seq = Any[]
        for i in 1:batchsize
            push!(tree_seq, data[p+i][3])
        end        
        
        p += batchsize
        push!(batch_data, (enc_seq, enc_len, tree_seq))
    end
    
    return batch_data
end

function minibatch2(data, batchsize)
    
    n = length(data)
    p = 0
    batch_data = Any[]
    
    if length(data) % batchsize != 0
        n = length(data)
        for i in 0:((length(data)%batchsize) - 1)
            insert!(data, n-i, data[n-i])
        end
    end
    
    while p + batchsize <= n
        
        max_seq_len = length(data[p+batchsize][1])
        enc_seq = zeros(Int64, batchsize, max_seq_len+2)
        enc_seq[:, max_seq_len+2] .= 2
        enc_len = Any[]
        
        for i in 1:batchsize
            seq = data[p+i][1]
            seq_len = length(seq)
            
            for j in 1:seq_len
                enc_seq[i, max_seq_len-seq_len+j+1] = seq[seq_len-j+1]
            end
            
            for k in 1:max_seq_len+1-seq_len #pad the start with start of seq tokens
                enc_seq[i,k] = 1
            end
            
            push!(enc_len, seq_len+2)
            
        end
                
        dec_seq = Any[]
        for i in 1:batchsize
            push!(dec_seq, data[p+i][2])
        end        
        
        tree_seq = Any[]
        for i in 1:batchsize
            push!(tree_seq, data[p+i][3])
        end        
        
        p += batchsize
        push!(batch_data, (enc_seq, enc_len, tree_seq))
    end
    
    return batch_data
end

mutable struct Tree
    parent
    num_children::Int
    children
end

Tree() = Tree(nothing, 0, Any[])

function add_child(t::Tree, c)
    if typeof(c) == Tree
        c.parent = t
    end
    push!(t.children, c)
    t.num_children = t.num_children + 1
end

function convert_to_tree(seq, i_left, i_right)
    t = Tree()
    level = 0
    left = 0
    for i in i_left:i_right
        if seq[i] == tok2int_output["("]
            if level == 0
                left = i
            end
            level = level + 1
        elseif seq[i] == tok2int_output[")"]
            level = level - 1
            if level == 0
                if i == left + 1
                    c = seq[i]
                else
                    c = convert_to_tree(seq, left + 1, i-1)
                end
                add_child(t, c)
            end
        elseif level == 0
            add_child(t, seq[i])
        end
    end
        
    return t
end

function to_list(t::Tree, tok2int) 
    r_list = []
    for i in 1:t.num_children
        if typeof(t.children[i]) == Tree
            push!(r_list, tok2int["("])
            cl = to_list(t.children[i], tok2int)
            for k in 1:length(cl)
                push!(r_list, cl[k])
            end
            push!(r_list, tok2int[")"])
        else
            push!(r_list, t.children[i])
        end
    end
    return r_list
end

function to_string(t::Tree, int2tok)
    
    r_list = []
    
    for i in 1:t.num_children
        if typeof(t.children[i]) == Tree
            push!(r_list, "( " * to_string(t.children[i], int2tok) * " )")
        else
            push!(r_list, int2tok[t.children[i]])
        end
    end
    
    return join(r_list, " ")
    
end

function norm_tree(r_list, tok2int, int2tok)
    
    q = [convert_to_tree(r_list, 1, length(r_list))]
    
    head = 1
    
    while head <= length(q)
        
        t = q[head]
        
        if (t.children[1] == tok2int["and"]) || (t.children[1] == tok2int["or"])
            
            # sort the following subchildren            
            k = []
            
            for i in 2:length(t.children) 
                
                if typeof(t.children[i]) == Tree
                    push!(k, (to_string(t.children[i], int2tok), i))
                else
                    push!(k, (string(t.children[i]), i))
                end
                
            end

            sorted_t_dict = []
            
            #k.sort(key=itemgetter(0))
            sort!(k, by = x -> x[1])
            
            for key1 in k
                push!(sorted_t_dict, t.children[key1[2]])
            end
            
            for i in 1:t.num_children-1
                t.children[i+1] = sorted_t_dict[i]
            end
            
        end
            
        # add children to q
        for i in 1:length(t.children)
                
            if typeof(t.children[i]) == Tree
                push!(q, t.children[i])
            end
            
        end                

        head = head + 1
        
    end
        
    return q[1]
    
end  

function mask(a, pad)
    a = copy(a)
    for i in 1:size(a, 1)
        j = size(a,2)
        while a[i, j] == pad && j > 1
            if a[i, j - 1] == pad
                a[i, j] = 0
            end
            j -= 1
        end
    end
    return a
end

function model_accuracy(model, data)
    total = 0
    num_sequences = 0
    for (x, y, t) in data
        
        num_sequences += 1
        
        y_pred = model(x) # x: Tx y: Ty y_pred: Ty'

        num_left_paren = sum([1 for c in y_pred if int2tok_output[c] == "("])    
        num_right_paren = sum([1 for c in y_pred if int2tok_output[c]== ")"])
                                
        diff = num_left_paren - num_right_paren
        if diff > 0
            for i in 1:diff
                push!(y_pred, tok2int_output[")"])
            end
        elseif diff < 0
            y_pred = y_pred[1:length(y_pred)+diff]
        end
        
        #norm_t = norm_tree(y_pred, tok2int_output, int2tok_output)
        norm_t = convert_to_tree(y_pred, 1, length(y_pred))
        y_pred = to_list(norm_t, tok2int_output)
        
        #norm_t = norm_tree(y, tok2int_output, int2tok_output)
        norm_t = convert_to_tree(y, 1, length(y))
        y = to_list(norm_t, tok2int_output)
        
        if length(y) == length(y_pred)
            if sum(y_pred .== y) == length(y)
                #println("MATCH")
                total += 1                    
            end
        end
    end
    return total / num_sequences
end

function rr()
    rng = Random.MersenneTwister(25);
    return rng
end

struct Embed
    w
end

function Embed(embedsize::Int, vocabsize::Int)
    w = rand(Uniform(-0.08,0.08), embedsize, vocabsize)
    w = Knet.Param(convert(Knet.KnetArray{Float32}, w))
    return Embed(w)
end
(e::Embed)(x) = e.w[:,x] #x: word id

struct Linear
    w
    b
    f
end

#Linear(i::Int,o::Int,f=identity) = Linear(param(o,i), param0(o), f)

function Linear(i::Int,o::Int,f=identity)
    w = rand(Uniform(-0.08,0.08), o, i)
    w = Knet.Param(convert(Knet.KnetArray{Float32},w))
    b = param0(o)
    return Linear(w, b, f)
end

(d::Linear)(x) = d.f.(d.w * mat(x,dims=1) .+ d.b)

struct seq2tree
    input_embed
    output_embed
    encoder
    decoder
    linear
    dropout
end

function seq2tree(X::Int, H::Int, Vq::Int, Va::Int, dropout::Real)
    a = Embed(X, Vq)
    b = Embed(X, Va)
    #Knet.seed!(25)
    c = RNN(X, H; rnnType=:lstm, numLayers=1, dropout=dropout, seed=42)
    #Knet.seed!(25)
    d = RNN(X, H; rnnType=:lstm, numLayers=1, dropout=dropout, seed=42)
    #d = RNN((X+H), H; rnnType=:lstm, numLayers=1, dropout=dropout)
    e = Linear(H,Va)
    f = dropout
    return seq2tree(a, b, c, d, e, f)
end

struct seq2treeAttn
    input_embed
    output_embed
    encoder
    decoder
    linear
    linear_att
    dropout
end

function seq2treeAttn(X::Int, H::Int, Vq::Int, Va::Int, dropout::Real)
    a = Embed(X, Vq)
    b = Embed(X, Va)
    #Knet.seed!(25)
    c = RNN(X, H; rnnType=:lstm, numLayers=1, dropout=dropout, seed=42)
    #Knet.seed!(25)
    d = RNN(X, H; rnnType=:lstm, numLayers=1, dropout=dropout, seed=42)
    #d = RNN((X+H), H; rnnType=:lstm, numLayers=1, dropout=dropout)
    e = Linear(H,Va)
    f = Linear(2*H, H, tanh)
    g = dropout
    return seq2treeAttn(a, b, c, d, e, f, g)
end
