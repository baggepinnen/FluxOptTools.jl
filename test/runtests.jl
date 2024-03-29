if haskey(ENV, "CI")
    ENV["PLOTS_TEST"] = "true"
    ENV["GKSwstype"] = "100" # gr segfault workaround
end

using FluxOptTools, Optim, Zygote, Flux, Plots, Test, Statistics, Random, StatsPlots
##
@testset "FluxOptTools" begin
@info "Testing FluxOptTools"
@testset "copy" begin
@info "Testing copy"

m = Chain(Dense(1,5,tanh), Dense(5,5,tanh) , Dense(5,1))
x = collect(LinRange{Float32}(-pi,pi,100)')
y = sin.(x)
sp = sortperm(x[:])

loss() = mean(abs2, m(x) .- y)
Zygote.refresh()
pars = Flux.params(m)
pars0 = deepcopy(pars)
npars = veclength(pars)
@test npars == 46

copy!(pars, zeros(pars))
@test all(all(iszero, p) for p in pars)

p = zeros(pars)
copy!(pars, 1:npars)
copy!(p, pars)
@test p == 1:npars

grads = Zygote.gradient(loss, pars)
grads0 = deepcopy(grads)

copy!(grads, zeros(grads))
@test all(all(iszero,grads[k]) for k in keys(grads.grads))

p = zeros(grads)
copy!(grads, 1:npars)
copy!(p, grads)
@test p == 1:npars
end

## Test optimization ============================================

@testset "optfuns" begin
@info "Testing optfuns"

model = Chain(Dense(1,5,tanh), Dense(5,5,tanh), Dense(5,1))
x = collect(LinRange{Float32}(-pi,pi,100)')
y = sin.(x)
sp = sortperm(x[:])

loss(m) = mean(abs2, m(x) .- y)
@show loss(model)

pars = Flux.params(model)
opt = Flux.setup(Adam(0.01), model)
for i = 1:500
    grads = Flux.gradient(model) do m
        loss(m)
    end
    Flux.update!(opt, model, grads[1])
end
adam_loss = loss(model)
@show loss(model)
@test loss(model) < 1e-1
plot(x[sp], [y[sp] model(x)[sp]])  |> display
contourf(() -> log10(1 + loss(model)), pars, color=:turbo, npoints=50, lnorm=1, seed=1234) |> display

lossfun, gradfun, fg!, p0 = optfuns(() -> loss(model), pars)
res = Optim.optimize(Optim.only_fg!(fg!), p0, BFGS())
@test loss(model) < 1e-3
@test loss(model) < adam_loss

contourf(() -> log10(1 + loss(model)), pars, color=:turbo, npoints=50, lnorm=1) |> display

plot(x[sp], [y[sp] model(x)[sp]]) |> display
end

@testset "benchmark" begin
@info "Benchmark Optim vs ADAM"

x = LinRange{Float32}(-pi, pi, 100)'
y = sin.(x)
loss(m) = mean(abs2, m(x) .- y)

@info "Run ADAM"
Zygote.refresh()
losses_adam = map(1:10) do i
    Random.seed!(i)
    model = Chain(Dense(1,5,tanh), Dense(5,5,tanh), Dense(5,1))
    pars = Flux.params(model)
    opt = Flux.Adam(0.2)
    trace = [loss(model)]
    t = @timed for i = 1:500
        l,back = Zygote.pullback(() -> loss(model), pars)
        push!(trace, l)
        grads = back(l)
        Flux.Optimise.update!(opt, pars, grads)
    end
    trace, t.time - t.gctime
end

@info "Run L-BFGS"
Zygote.refresh()
res_lbfgs = map(1:10) do i
    Random.seed!(i)
    model = Chain(Dense(1,5,tanh), Dense(5,5,tanh), Dense(5,1))
    pars = Flux.params(model)
    lossfun, gradfun, fg!, p0 = optfuns(() -> loss(model), pars)
    res = @timed Optim.optimize(Optim.only_fg!(fg!), p0, BFGS(), Optim.Options(iterations=500, store_trace=true))
    res.value, res.time - res.gctime
end

@info "Run SL-BFGS"
Zygote.refresh()
losses_SLBFGS = map(1:10) do i
    Random.seed!(i)
    model = Chain(Dense(1,5,tanh), Dense(5,5,tanh), Dense(5,1))
    pars = Flux.params(model)
    lossfun, gradfun, fg!, p0 = optfuns(() -> loss(model), pars)
    opt = SLBFGS(lossfun, p0; m=15, ᾱ=0.85, ρ=false, λ=0.0005, κ=0.2)
    function train(opt, p0, iters=20)
        p = copy(p0)
        g = zeros(veclength(pars))
        trace = [loss(model)]
        for i = 1:iters
            g = gradfun(g,p)
            p = apply(opt, g, p)
            push!(trace, opt.fold)
        end
        trace
    end
    trace = @timed train(opt,p0, 500)
    trace.value, trace.time - trace.gctime
end

# Loss plot
valuetrace(r) = getfield.(r.trace, :value)
valuetraces = valuetrace.(first.(res_lbfgs))
plot(valuetraces[1], yscale=:log10, lab="LBFGS", c=:red, xlabel="Epochs", ylabel="Loss", yticks=[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
plot!(valuetraces[2:end], lab="", c=:red)
median_loss_BFGS = median(last.(valuetraces))

SLBFGS_loss = first.(losses_SLBFGS)
plot!(SLBFGS_loss[1], lab="SLBFGS", c=:green)
plot!(SLBFGS_loss[2:end], lab="", c=:green)
median_loss_SLBFGS = median(last.(SLBFGS_loss))

ADAM_loss = first.(losses_adam)
plot!(ADAM_loss[1], lab="ADAM", c=:blue)
plot!(ADAM_loss[2:end], lab="", c=:blue) |> display
median_loss_ADAM = median(last.(ADAM_loss))

@test median_loss_ADAM > median_loss_SLBFGS > median_loss_BFGS

# Time plot
median_LBFGS = median(last.(res_lbfgs)[2:end])
median_ADAM = median(last.(losses_adam)[2:end])
median_SLBFGS = median(last.(losses_SLBFGS)[2:end])
time_matrix = reduce(hcat, [last.(res_lbfgs)[2:end], last.(losses_SLBFGS)[2:end], last.(losses_adam)[2:end]])
time_matrix ./= median_ADAM
boxplot(["LBFGS"], time_matrix[:, 1], legend=false, ylabel="Relative time", c=:red)
boxplot!(["SLBFGS"], time_matrix[:, 2], legend=false, c=:green)
boxplot!(["ADAM"], time_matrix[:, 3], legend=false, c=:blue) |> display

@test median_ADAM < median_SLBFGS < median_LBFGS
end

end
