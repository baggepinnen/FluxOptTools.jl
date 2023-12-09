##
using LinearAlgebra, Optim, Zygote, RecipesBase, Random
using Lux, ComponentArrays, Flux.Losses, Statistics, Optimisers
##
function optfuns(Lux_loss, pars)
    ##
    p0 = zeros(eltype(pars), size(pars))
    copy!(p0, pars)
    ##
    (l, st_), back = Zygote.pullback(p -> Lux_loss(x, y, model, p, st), re(pars))
    gs = back((one(l), nothing))[1]
    grads, _ = Optimisers.destructure(gs)
	##
    gradfun = function (g,w)
        copy!(pars, w)
        (l, st_), back = Zygote.pullback(p -> Lux_loss(x, y, model, p, st), re(pars))
        gs = back((one(l), nothing))[1]
        copy!(g, first(Optimisers.destructure(gs)))
    end
    lossfun = function (w)
        copy!(pars, w)
        Lux_loss(x, y, model, p, st)
    end
    fg! = function (F,G,w)
        copy!(pars, w)
        if !isnothing(G)
            (l, st_), back = Zygote.pullback(p -> Lux_loss(x, y, model, p, st), re(pars))
            gs = back((one(l), nothing))[1]
            copy!(G, first(Optimisers.destructure(gs)))
            return l
        end
        if !isnothing(F)
            return Lux_loss(x, y, model, p, st)
        end
    end
    lossfun, gradfun, fg!, p0
end
##
model = Lux.Chain(Lux.Dense(20 => 20), Lux.Dense(20 => 20))
ps, st = Lux.setup(Random.default_rng(), model)
ps = ComponentArray(ps)
x = randn(20, 2)
y = randn(20, 2)
##
function Lux_loss(x, y, model, ps, st) 
    ŷ, st_ = model(x, ps, st)
    return mse(ŷ, y), st_
end
##
Zygote.refresh()
##
pars, re = Optimisers.destructure(ps)
##
lossfun, gradfun, fg!, p0 = optfuns(Lux_loss, pars)
##
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))
##
