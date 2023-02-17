module FluxOptTools

using LinearAlgebra, Optim, Flux, Zygote, RecipesBase

export veclength, optfuns

veclength(grads::Zygote.Grads) = sum(length(grads[p]) for p in grads.params)
veclength(params::Flux.Params) = sum(length, params.params)
veclength(x) = length(x)
Base.zeros(grads::Zygote.Grads) = zeros(veclength(grads))
Base.zeros(pars::Flux.Params) = zeros(veclength(pars))

function optfuns(loss, pars::Union{Flux.Params,Zygote.Params})
    grads = Zygote.gradient(loss, pars)
    p0 = zeros(pars)
    copy!(p0, pars)
    gradfun = function (g, w)
        copy!(pars, w)
        grads = Zygote.gradient(loss, pars)
        copy!(g, grads)
    end
    lossfun = function (w)
        copy!(pars, w)
        loss()
    end
    fg! = function (F, G, w)
        copy!(pars, w)
        if G != nothing
            l, back = Zygote.pullback(loss, pars)
            grads = back(1)
            copy!(G, grads)
            return l
        end
        if F != nothing
            return loss()
        end
    end
    lossfun, gradfun, fg!, p0
end

@recipe function lossplot(loss::Function, pars::Flux.Params; lnorm=0.1, npoints=30)
    p = zeros(pars)
    copy!(p, pars)
    pcopy = deepcopy(p)
    n0 = norm(p)
    dx, dy = randn(length(p)), randn(length(p))
    dx *= n0 * lnorm / norm(dx)
    dy *= n0 * lnorm / norm(dy)
    pertvec = LinRange(-1, 1, npoints)
    losses = map(Iterators.product(pertvec, pertvec)) do (lx, ly)
        pi = p + lx * dx + ly * dy
        copy!(pars, pi)
        loss()
    end
    copy!(pars, pcopy)
    seriestype --> :contour
    pertvec, pertvec, losses
end

include("SLBFGS.jl")

end # module
