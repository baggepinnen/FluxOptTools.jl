export SLBFGS
using Distributions, OnlineStats
Base.@kwdef mutable struct SLBFGS{F,Σ}
    m::Int = 20# Memory length, was ∈ [2, 54] in paper
    λ::Float64 = 1. # regularizer, was ∈ [0.04, 1] in paper
    ρ::Bool = false # Algorithm indicator
    κ::Float64 = 0.5 # Step-length scaling factor ∈ (0,1)
    ᾱ::Float64 = 1. # maximum allowed step-length ∈ (0,1)  TODO: can be function of k
    α::Float64 = ᾱ
    fold::Float64 = Inf
    k::Int = 1 # Internal iteration index
    σ²::Σ = OnlineStats.Variance()
    xold::Vector{Float64}
    gold::Vector{Float64}
    f::F
    Y::Matrix{Float64}
    S::Matrix{Float64}
end

function SLBFGS(f,x0; m=20, kwargs...)
    d = length(x0)
    Y = zeros(d,m)
    S = zeros(d,m)
    SLBFGS(;m=m, xold=x0, gold=zeros(d), f=f, Y=Y, S=S, kwargs...)
end


function apply(opt::SLBFGS, g, x)
    ρ, κ, ᾱ = opt.ρ, opt.κ, opt.ᾱ
    update_YS!(opt, g, x)
    p = search_direction(opt, g)
    while true
        ξ = x + opt.α*p
        c,fξ = acceptance_indicator(opt, ξ)
        # Update vars
        if c
            x = ξ
            opt.fold = fξ
        end
        opt.α = c*(1/opt.k)^ρ * ᾱ + (1-c)κ*opt.α
        opt.k += 1
        c && break
    end
    x
end

function update_YS!(opt, g, x)
    Y,S = opt.Y, opt.S
    ind = ((opt.k-1) % opt.m) + 1
    Y[:,ind] .= g .- opt.gold
    S[:,ind] .= x .- opt.xold
    opt.gold  = g
    opt.xold  = x
end

function search_direction(opt, g)
    λ = opt.λ
    Y,S = opt.Y, opt.S
    # r4 = R1'\Y1'y
    # r5 = √(λ + y'y - r4'r4)
    # r6 = (y'Y2 - r4'R2)/r5
    # # R6TR6 = R4'R4 - r6'r6 + r3'r3 # TODO: one is really interested in R6 and this should be obtained by downdating rith r6'r6 and updating with r3'r3
    # R6 = downdate(R4, r6)
    # R6 = update(R6, r3)
    R = cholesky(Hermitian(Y'Y + λ*I))
    H̄ = I
    z = inv(λ)Y*(S'g) + H̄*g # QUESTION: they do not mention what H̄ they use
    w = R\(R.L'\(Y'z))
    p = -z + Y*w
    β = 2*max(0,p'g/(g'g))
    @. p -= β*g

end

function acceptance_indicator(opt, ξ)
    fξ = opt.f(ξ)
    ϵ = fξ - opt.fold
    ϵ < 0 && return (true, fξ)
    r = rand()
    # cdf(Normal(-ϵ,σ²)) # QUESTION: use online stats to calculate this variance?
    fit!(opt.σ²,fξ)
    r < cdf(Normal(0,√(value(opt.σ²))), -ϵ), fξ # QUESTION: The paper is not clear about this acceptance probability, this is what I think they mean, but should really ask the authors to clarify
end

##
using FluxOptTools, Optim, Zygote, Flux, Plots, Test, Statistics, Random
m = Chain(Dense(1,3,tanh) , Dense(3,1))
x = LinRange(-pi,pi,100)'
y = sin.(x)
sp = sortperm(x[:])

loss() = mean(abs2, m(x) .- y)

@show loss()
# Zygote.refresh()
pars = Flux.params(m)

lossfun, gradfun, fg!, p0 = optfuns(loss, pars)

opt = SLBFGS(lossfun,p0; m=3, ᾱ=0.2, ρ=false, λ=10.)
function train(opt, p0, iters=20)
    p = copy(p0)
    g = zeros(veclength(pars))
    trace = [loss()]
    for i = 1:iters
        g = gradfun(g,p)
        p = apply(opt, g, p)
        i % 20 == 0 && @show opt.fold
        push!(trace, opt.fold)
    end
    trace
end

trace = train(opt,p0, 1000)
plot(trace, yscale=:log10, xscale=:log10, size=(400,300))
# plot(x', [y' m(x)'])  |> display
