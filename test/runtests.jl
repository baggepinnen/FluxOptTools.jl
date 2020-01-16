using FluxOptTools, Optim, Zygote, Flux, Plots, Test, Statistics, Random
##
@testset "FluxOptTools" begin
@info "Testing FluxOptTools"
@testset "copyto" begin
@info "Testing copyto"

m = Chain(Dense(1,5,tanh), Dense(5,5,tanh) , Dense(5,1))
x = LinRange(-pi,pi,100)'
y = sin.(x)
sp = sortperm(x[:])

loss() = mean(abs2, m(x) .- y)
Zygote.refresh()
pars = Flux.params(m)
pars0 = deepcopy(pars)
npars = veclength(pars)
@test npars == 46

copyto!(pars, zeros(pars))
@test all(all(iszero, p) for p in pars)

p = zeros(pars)
copyto!(pars, 1:npars)
copyto!(p, pars)
@test p == 1:npars

grads = Zygote.gradient(loss, pars)
grads0 = deepcopy(grads)

copyto!(grads, zeros(grads))
@test all(all(iszero,grads[k]) for k in keys(grads.grads))

p = zeros(grads)
copyto!(grads, 1:npars)
copyto!(p, grads)
@test p == 1:npars
end

## Test optimization ============================================

end


# NOTE: tests below fail if they are in a testset, probably Zygote's fault

m = Chain(Dense(1,5,tanh), Dense(5,5,tanh) , Dense(5,1))
x = LinRange(-pi,pi,100)'
y = sin.(x)
sp = sortperm(x[:])

loss() = mean(abs2, m(x) .- y)
@show loss()
Zygote.refresh()
pars = Flux.params(m)

opt = ADAM(0.01)
@show loss()
for i = 1:500
    grads = Zygote.gradient(loss, pars)
    Flux.Optimise.update!(opt, pars, grads)
end
@show loss()
@test loss() < 1e-1
plot(x[sp], [y[sp] m(x)[sp]])  |> display
plot(loss, pars, l=0.5, npoints=50, seriestype=:contour) |> display

lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
res = Optim.optimize(Optim.only_fg!(fg!), p0, BFGS())
@test loss() < 1e-3
plot(loss, pars, l=0.1, npoints=50) |> display

plot(x[sp], [y[sp] m(x)[sp]]) |> display



## Benchmark Optim vs ADAM
losses_adam = map(1:10) do i
    @show i
    Random.seed!(i)
    m = Chain(Dense(1,5,tanh), Dense(5,5,tanh) , Dense(5,1))
    x = LinRange(-pi,pi,100)'
    y = sin.(x)
    loss() = mean(abs2, m(x) .- y)
    Zygote.refresh()
    pars = Flux.params(m)
    opt = Flux.ADAM(0.2)
    trace = [loss()]
    for i = 1:500
        l,back = Zygote.pullback(loss, pars)
        push!(trace, l)
        grads = back(l)
        Flux.Optimise.update!(opt, pars, grads)
    end
    trace
end

res_lbfgs = map(1:10) do i
    @show i
    Random.seed!(i)
    m = Chain(Dense(1,5,tanh), Dense(5,5,tanh) , Dense(5,1))
    x = LinRange(-pi,pi,100)'
    y = sin.(x)
    loss() = mean(abs2, m(x) .- y)
    Zygote.refresh()
    pars = Flux.params(m)
    lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
    res = Optim.optimize(Optim.only_fg!(fg!), p0, BFGS(), Optim.Options(iterations=500, store_trace=true))
    res
end

losses_SLBFGS = map(1:10) do i
    @show i
    Random.seed!(i)
    m = Chain(Dense(1,5,tanh), Dense(5,5,tanh) , Dense(5,1))
    x = LinRange(-pi,pi,100)'
    y = sin.(x)
    loss() = mean(abs2, m(x) .- y)
    Zygote.refresh()
    pars = Flux.params(m)
    lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
    opt = SLBFGS(lossfun,p0; m=3, ᾱ=1., ρ=false, λ=.0001, κ=0.1)
    function train(opt, p0, iters=20)
        p = copy(p0)
        g = zeros(veclength(pars))
        trace = [loss()]
        for i = 1:iters
            g = gradfun(g,p)
            p = apply(opt, g, p)
            push!(trace, opt.fold)
        end
        trace
    end
    trace = train(opt,p0, 500)
end


##
valuetrace(r) = getfield.(r.trace, :value)
valuetraces = valuetrace.(res_lbfgs)
plot(valuetraces, yscale=:log10, xscale=:identity, lab="", c=:red)
plot!(losses_adam, lab="", c=:blue, xlabel="Epochs", ylabel="Loss")
plot!(losses_SLBFGS, lab="", c=:green)
