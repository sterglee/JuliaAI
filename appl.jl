
using Flux
using Main.TransformersLite

mha = TransformersLite.MultiHeadAttention(4, 32, 32)
Flux._big_show(stdout, mha)
#=
TransformersLite.MultiHeadAttention(
4,
Dense(32 => 32; bias=false),          # 1_024 parameters
Dense(32 => 32; bias=false),          # 1_024 parameters
Dense(32 => 32; bias=false),          # 1_024 parameters
Dense(32 => 32),                      # 1_056 parameters
)                   # Total: 5 arrays, 4_128 parameters, 16.422 KiB.
=#


x = randn(Float32, 32, 20, 2) # d×n×B
mask = make_causal_mask(ones(32, 20))
y, scores = mha(x, x, x; mask=mask) # 32×20×2 Array{Float32, 3}, 20×20×4×2 Array{Float32, 4}

using Flux
loss = sum # dummy loss function
grads = Flux.gradient(m -> loss(m(x, x, x; mask=mask)[1]), mha)
keys(grads[1]) # (:nhead, :denseQ, :denseK, :denseV, :denseO)
struct TransformerBlock{
    MHA<:TransformersLite.MultiHeadAttention,
    N1<:LayerNorm,
    D1<:Dense,
    D2<:Dense,
    N2<:LayerNorm,
    DO<:Dropout}
multihead_attention::MHA
norm_attention::N1
dense1::D1
dense2::D2
norm_feedforward::N2
dropout::DO
end

Flux.@layer TransformerBlock # make whole layer trainable



TransformerBlock(
    nhead::Int,
    dim_model::Int,
    dim_hidden::Int;
    act=relu,
    pdrop::Float64=0.1,
    ) = TransformerBlock(
        TransformersLite.MultiHeadAttention(nhead, dim_model, dim_model),
        LayerNorm(dim_model),
        Dense(dim_model, dim_hidden, act),
        Dense(dim_hidden, dim_model),
        LayerNorm(dim_model),
        Dropout(pdrop),
        )


    function (t::TransformerBlock)(x::A; mask::M=nothing) where {
        A<:AbstractArray, M<:Union{Nothing, AbstractArray{Bool}}}
    h, scores = t.multihead_attention(x, x, x; mask=mask) # (dm, N, B)
    h = t.dropout(h)
    h = x + h
    h = t.norm_attention(h)            # (dm, N, B)
    hff = t.dense1(h)                  # (dh, N, B)
    hff = t.dense2(hff)                # (dm, N, B)
    hff = t.dropout(hff)
    h = h + hff
    h = t.norm_feedforward(h)          # (dm, N, B)
    h
    end


    block = TransformerBlock(4, 32, 32*4)

    x = randn(Float32, 32, 20, 2) # d×n×B
    mask = make_causal_mask(ones(32, 20)) # 20×20 Matrix{Bool}
    y = block(x; mask=mask) # 32×20×2 Array{Float32, 3}

    loss = sum # dummy loss function
    grads = Flux.gradient(m -> loss(m(x; mask=mask)), block)
    keys(grads[1]) # (:multihead_attention, :norm_attention, :dense1, :dense2, :norm_feedforward, :dropout)





    struct TransformerGenerator{
        E<:Flux.Embedding,
        PE<:Flux.Embedding,
        DO<:Dropout,
        TB<:Vector{<:TransformerBlock},
        D<:Dense,
        M<:Union{Nothing, AbstractMatrix{Bool}},
        }
    embedding::E
    position_encoding::PE
    dropout::DO
    blocks::TB
    head::D
    mask::M # optional buffer
    end

    Flux.@layer TransformerGenerator trainable=(embedding, position_encoding, blocks, dropout, head)



    function (t::TransformerGenerator)(x::A; mask::M=t.mask) where {
        A<:AbstractArray, M<:Union{Nothing, AbstractMatrix{Bool}}}
    x = t.embedding(x)              # (dm, N, B)
    N = size(x, 2)
    x = x .+ t.position_encoding(1:N) # (dm, N, B)
    x = t.dropout(x)                # (dm, N, B)
    for block in t.blocks
        x = block(x; mask=mask)     # (dm, N, B)
    end
    x = t.head(x)                   # (vocab_size, N, B)
    x
    end


        context_size = 64
        dim = 32
        nheads = 4
        vocab_size = 71
        mask = make_causal_mask(ones(context_size, context_size))
        model = TransformerGenerator(
            Embedding(vocab_size => dim),
            Embedding(context_size => dim),
            Dropout(0.1),
            TransformerBlock[
                TransformerBlock(4, dim, dim * 4; pdrop=0.1),
                TransformerBlock(4, dim, dim * 4; pdrop=0.1),
                TransformerBlock(4, dim, dim * 4; pdrop=0.1),
                ],
            Dense(dim, vocab_size),
            copy(mask)
            )
            Flux._big_show(stdout, model)


            x = reshape(rand(1:vocab_size, 34), :, 1) # make it a batch of 1
            mask = make_causal_mask(ones(dim, length(x)))
            y = model(x; mask=mask) # 71×34×1 Array{Float32, 3}


            X = rand(1:vocab_size, 34, 10)
            Y = model(X; mask=mask) # 71×34×10

            function tail(A::AbstractMatrix, n::Int)
                n = min(n, size(A, 1))
                A[(end - n + 1):end, :]
            end


            using Random, StatsBase
            function tgenerate(
                rng::AbstractRNG, model::TransformerGenerator, context::AbstractMatrix{T}
                ; context_size::Int, max_tokens::Int=100,
                ) where T
            for i in 1:max_tokens
                context_crop = tail(context, context_size)
                n = size(context_crop, 1)
                mask = isnothing(model.mask) ? nothing : view(model.mask, 1:n, 1:n)
                logits = model(context_crop; mask=mask) |> cpu # (vocab_size, n, B)
                logits = logits[:, end, :] # (vocab_size, B)
                context_next = multinomial_sampling(rng, logits)
                context = cat(context, context_next; dims=1)
            end
            context
            end

            function tgenerate(model::TransformerGenerator, context::AbstractMatrix; kwargs...)
                tgenerate(Random.default_rng(), model, context; kwargs...)
            end

            function multinomial_sampling(rng::AbstractRNG, logits::AbstractMatrix)
                probs = softmax(logits; dims=1)
                tokens = [sample(rng, Weights(p)) for p in eachcol(probs)]
                    tokens
                end

                struct tIndexTokenizer{T}
                    vocabulary::Vector{T}
                    lookup::Dict{T, Int}
                    unksym::T
                    unkidx::Int
                    function tIndexTokenizer(vocab::Vector{T}, unksym::T) where T
                        if !(unksym ∈ vocab)
                            pushfirst!(vocab, unksym)
                            unkidx = 1
                        else
                            unkidx = findfirst(isequal(unksym), vocab)
                        end
                        lookup = Dict(x => idx for (idx, x) in enumerate(vocab))
                            new{T}(vocab, lookup, unksym, unkidx)
                        end
                    end

                    Base.length(tokenizer::tIndexTokenizer) = length(tokenizer.vocabulary)

                    function Base.show(io::IO, tokenizer::tIndexTokenizer)
                        T = eltype(tokenizer.vocabulary)
                        print(io, "tIndexTokenizer{$(T)}(length(vocabulary)=$(length(tokenizer)), unksym=$(tokenizer.unksym))")
                    end

                context = reshape([1], 1, 1) # start with the new line symbol
                out = tgenerate(model, context; context_size=64) # 101×1 Matrix{Int64}














