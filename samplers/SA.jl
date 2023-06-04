"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com
"

module SA
    include("tools.jl")
    using ..SamplingTools: hj_info, default_temp_range, generate_nbr_dict,
                            generate_nbr, discrete_aggregation, add_noise,
                            add_cross_talk, hj_info!, generate_nbr!
    using Random
    using Distributed
    using Profile
    using BenchmarkTools
    using LinearAlgebra
    using SharedArrays
    function anneal!(spin_vector::Array{Float64, 1},
                    nbs_idxs::Array{Int64, 2},
                    nbs_vals::Array{Float64, 2},
                    temp_schedule::Array{Float64, 1})
        maxnb = size(nbs_idxs, 1)
        nsteps = size(temp_schedule, 1)
        nspins = size(spin_vector, 1)
        perm = collect(1:nspins)
        istep = 1
        temp = 0.0
        ispin = 1
        spin_nb = 1
        deg = 1
        ediff =  0.0
        spin = 1
        idx = 1

        @fastmath @inbounds begin
            for istep = 1:nsteps
                temp = temp_schedule[istep]
                randperm!(perm)
                for idx = 1:nspins
                    ispin = perm[idx]
                    ediff = 0.0
                    spin = spin_vector[ispin]
                    for deg = 1:maxnb
                        spin_nb = nbs_idxs[deg, ispin]
                        if spin_nb === ispin
                            ediff += -2.0*spin*nbs_vals[deg, ispin]
                        elseif spin_nb !== 0
                            ediff += -2.0*spin*(nbs_vals[deg, ispin]*spin_vector[spin_nb])
                        end
                    end
                    if ediff <= 0.0
                        spin_vector[ispin] *= -1.0
                    elseif exp(-1.0 * ediff / temp) > rand()
                        spin_vector[ispin] *= -1.0
                    end
                end
            end
        end
        # return spin_vector
    end


    function sample(h::Dict, J::Dict; num_reads=1000, num_steps=1000,
                    add_xtalk=false, noisy=false,
                    temp_schedule=zeros(Float64, 10),
                    initial_state=Array{Float64, 1}(), return_raw=false)
        # forward anneal usage only
        nspins = length(h)
        

        # prepare spin vectors
        if length(initial_state) == nspins
            @assert all(abs.(initial_state) .== 1.0)
            spin_vectors = [deepcopy(initial_state) for i =1:num_reads]
        else
            spin_vectors = rand([-1.0, 1.0], nspins, num_reads)
            spin_vectors = [spin_vectors[:, i] for i =1:num_reads]
        end

        JM, M, degs = hj_info(h, J)
        # allow for custom temperature schedules, but if none given then
        # one will be calculated according to the num_steps
        if all(temp_schedule .== 0)
            hot_t, cold_t = default_temp_range(JM)
            temp_sch = collect(LinRange(hot_t, cold_t, num_steps))
        end

        # generate index and value array of problem
        nbs_idxs, nbs_vals = generate_nbr(nspins, M, maximum(degs) + 1)
        Js = copy(JM)

        if !noisy & !add_xtalk
            # standard anneal
            @inbounds for i = 1:num_reads
                anneal!(spin_vectors[i], nbs_idxs, nbs_vals, temp_schedule)
            end
        elseif noisy & add_xtalk
            # anneal with noise and x-talk, need to re-egenrate the problem
            # upon each read due to the change of parameter
            @inbounds for i = 1:num_reads
                hc, jc = add_cross_talk(add_noise(h, J)...)
                hj_info!(Js, M, degs, hc, jc)
                generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                anneal!(spin_vectors[i], nbs_idxs, nbs_vals, temp_schedule)
            end
        elseif add_xtalk & !noisy
            # anneal with x-talk, need to re-egenrate the problem
            # once before the runs as problem is the same consistently
            hc, jc = add_cross_talk(h, J)
            hj_info!(Js, M, degs, hc, jc)
            generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
            @inbounds for i = 1:num_reads
                anneal!(spin_vectors[i], nbs_idxs, nbs_vals, temp_schedule)
            end
        else
            # anneal with noise, need to re-egenrate the problem
            # upon each read due to the change of parameters
            @inbounds for i = 1:num_reads
                hc, jc = add_noise(h, J)
                hj_info!(Js, M, degs, hc, jc)
                generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                anneal!(spin_vectors[i], nbs_idxs, nbs_vals, temp_schedule)
            end
        end
        if return_raw
            return spin_vectors
        end
        return discrete_aggregation(spin_vectors, num_reads, JM)
    end

    export sample, anneal!
end
