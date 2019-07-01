module FluxOptTools

using LinearAlgebra, Optim, Flux, Zygote, RecipesBase
import Base.copyto!

export gradlength, paramlength, optfuns

gradlength(grads::Zygote.Grads) = sum(length(g[1]) for g in grads.grads)
paramlength(params::Flux.Params) = sum(length, params.params)
Base.zeros(grads::Zygote.Grads) = zeros(gradlength(grads))
Base.zeros(pars::Flux.Params) = zeros(paramlength(pars))

# Grads =============================================

function copyto!(v::AbstractArray, grads::Zygote.Grads)
    @assert length(v) == gradlength(grads)
    s = 1
    for g in grads.grads
        l = length(g[2])
        v[s:s+l-1] .= vec(g[2])
        s += l
    end
    v
end

function copyto!(grads::Zygote.Grads, v::AbstractArray)
    s = 1
    for g in grads.grads
        l = length(g[2])
        g[2] .= reshape(v[s:s+l-1], size(g[2]))
        s += l
    end
    grads
end

# Params =============================================

function copyto!(v::AbstractArray, pars::Flux.Params)
    @assert length(v) == paramlength(pars)
    s = 1
    for g in pars.params
        l = length(g)
        v[s:s+l-1] .= vec(g)
        s += l
    end
    v
end

function copyto!(pars::Flux.Params, v::AbstractArray)
    s = 1
    for p in pars.params
        l = length(p)
        p .= reshape(v[s:s+l-1], size(p))
        s += l
    end
    pars
end


function optfuns(loss, pars::Flux.Params)
    grads = Zygote.gradient(loss, pars)
    p0 = copyto!(zeros(pars), pars)
    gradfun = function (g,w)
        copyto!(pars, w)
        grads = Zygote.gradient(loss, pars)
        copyto!(g, grads)
    end
    lossfun = function (w)
        copyto!(pars, w)
        loss()
    end
    fg! = function (F,G,w)
        copyto!(pars, w)
        if G != nothing
            l, back = Zygote.forward(loss, pars)
            grads = back(l)
            copyto!(G, grads)
            return l
        end
        if F != nothing
            return loss()
        end
    end
    lossfun, gradfun, fg!, p0
end

@recipe function lossplot(loss::Function, pars::Flux.Params; l=0.1, npoints=30)
    p       = zeros(pars)
    copyto!(p,pars)
    pcopy   = deepcopy(p)
    n0      = norm(p)
    dx,dy   = randn(length(p)),randn(length(p))
    dx     *= n0*l/norm(dx)
    dy     *= n0*l/norm(dy)
    pertvec = LinRange(-1,1,npoints)
    losses = map(Iterators.product(pertvec,pertvec)) do (lx,ly)
        pi = p + lx*dx + ly*dy
        copyto!(pars, pi)
        loss()
    end
    copyto!(pars, pcopy)
    seriestype --> :contour
    pertvec,pertvec,losses
end


end # module
