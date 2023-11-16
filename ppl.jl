using Distributions
using SplittableRandoms

## Smallest PPL!!

const current_log_likelihood = Ref(0.0)

function observe(observation, distribution)
    current_log_likelihood[] += logpdf(distribution, observation)
end

function posterior(rng, probabilistic_program, n_particles)
    samples = Float64[] 
    log_weights = Float64[]

    for i in 1:n_particles 
        current_log_likelihood[] = 0.0
        push!(samples, probabilistic_program(rng))
        push!(log_weights, current_log_likelihood[])
    end

    return sum(samples .* exponentiate_normalize(log_weights))
end


### Example

function my_first_probabilistic_program(rng)
    coin_index = rand(rng, DiscreteUniform(0, 2)) 
    for i in 1:3 
        observe(1, Bernoulli(coin_index / 3))
    end
    return coin_index == 1 ? 1 : 0
end


### Utils

function exponentiate_normalize(vector)
    exponentiated = exp.(vector .- maximum(vector))
    return exponentiated / sum(exponentiated)
end


### Run demo..

rng = SplittableRandom(1)

println("Analytical: ", (1/24) / (1/24 + 1/3))
println("        MC: ", posterior(rng, my_first_probabilistic_program, 1_000_000))



nothing