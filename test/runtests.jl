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
npars = Optim4Flux.paramlength(pars)
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
plot(loss, pars,0.1) |> display

lossfun, gradfun = optfuns(loss, pars)
res = Optim.optimize(lossfun, gradfun, randn(paramlength(pars)))
@test loss() < 1e-3
plot(loss, pars,0.1) |> display

plot(x[sp], [y[sp] m(x)[sp]]) |> display
