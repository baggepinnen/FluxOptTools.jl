using FluxOptTools, Optim, Zygote, Flux, Test
##
m = Chain(Dense(1,3,tanh) , Dense(3,1))
x = LinRange(-pi,pi,100)'
y = sin.(x)
sp = sortperm(x[:])

loss() = mean(abs2, m(x) .- y)
Zygote.refresh()
pars = Flux.params(m)
pars0 = deepcopy(pars)
npars = paramlength(pars)
@test npars == 10
@test begin
    copyto!(pars, zeros(pars))
    all(all(iszero, p) for p in pars)
end
@test begin
    p = zeros(pars)
    copyto!(pars, 1:npars)
    copyto!(p, pars)
    p == 1:npars
end
grads = Zygote.gradient(loss, pars)
grads0 = deepcopy(grads)
@test begin
    copyto!(grads, zeros(grads))
    all(all(iszero,grads[k]) for k in keys(grads.grads))
end
@test begin
    p = zeros(grads)
    copyto!(grads, 1:npars)
    copyto!(p, grads)
    p == 1:npars
end


## Test optimization ============================================

m = Chain(Dense(1,3,tanh) , Dense(3,1))
x = LinRange(-pi,pi,100)'
y = sin.(x)
sp = sortperm(x[:])

loss() = mean(abs2, m(x) .- y)
@show loss()
Zygote.refresh()
pars = Flux.params(m)

opt = ADAM(0.01)
cb = ()-> @show loss()
for i = 1:500
    grads = Zygote.gradient(loss, pars)
    Flux.Optimise.update!(opt, pars, grads)
    @show loss()
end
@test loss() < 1e-1
plot(x[sp], [y[sp] m(x)[sp]])  |> display
plot(loss, pars, l=0.5, npoints=50, seriestype=:contour) |> display

lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
res = Optim.optimize(Optim.only_fg!(fg!), p0)
@test loss() < 1e-3
plot(loss, pars, l=0.1, npoints=50) |> display

plot(x[sp], [y[sp] m(x)[sp]]) |> display



## Benchmark Optim vs ADAM
losses_adam = map(1:20) do i
    @show i
    Random.seed!(i)
    m = Chain(Dense(1,3,tanh) , Dense(3,1))
    x = LinRange(-pi,pi,100)'
    y = sin.(x)
    loss() = mean(abs2, m(x) .- y)
    Zygote.refresh()
    pars = Flux.params(m)
    opt = ADAM(0.02)
    trace = [loss()]
    for i = 1:1000
        l,back = Zygote.forward(loss, pars)
        push!(trace, l)
        grads = back(l)
        Flux.Optimise.update!(opt, pars, grads)
    end
    trace
end

res_lbfgs = map(1:20) do i
    @show i
    Random.seed!(i)
    m = Chain(Dense(1,3,tanh) , Dense(3,1))
    x = LinRange(-pi,pi,100)'
    y = sin.(x)
    loss() = mean(abs2, m(x) .- y)
    Zygote.refresh()
    pars = Flux.params(m)
    lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
    res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))
    res
end


##
valuetrace(r) = getfield.(r.trace, :value)
valuetraces = valuetrace.(res_lbfgs)
plot(valuetraces, yscale=:log10, xscale=:identity, lab="", c=:red)
plot!(losses_adam, lab="", c=:blue, xlabel="Epochs", ylabel="Loss")
