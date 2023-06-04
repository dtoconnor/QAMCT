"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com
Based off the work published in
https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.15.014029
T. Albash and J. Marshall, Comparing relaxation mechanisms in
quantum and classical transverse-field annealing,
Physical Review Applied15, 014029 (2021),arXiv:2009.04934.
"

module SVMCTF
    include("tools.jl")
    using ..SamplingTools: hj_info, default_temp_range, generate_nbr_dict,
                            generate_nbr, rotor_aggregation, add_noise,
                            add_cross_talk, hj_info!, generate_nbr!
    using Random
    using Distributed
    using LinearAlgebra
    using SharedArrays

    function anneal!(spin_vector::Array{Float64, 1},
                    nbs_idxs::Array{Int64, 2},
                    nbs_vals::Array{Float64, 2},
                    a_schedule::Array{Float64, 1},
                    # must be same size or will crash from @inbounds
                    b_schedule::Array{Float64, 1},
                    temp::Float64)
        maxnb = size(nbs_idxs, 1)
        nspins = size(spin_vector, 1)
        perm = collect(1:nspins)
        istep = 1
        z_mag = 0.0
        a_field = 0.0
        b_field = 0.0
        ispin = 1
        spin_nb = 1
        deg = 1
        ediff =  0.0
        theta = 0.0
        idx = 1

        @fastmath @inbounds begin
            for istep in eachindex(a_schedule)
                a_field = a_schedule[istep]
                b_field = b_schedule[istep]
                randperm!(perm)
                for idx in eachindex(perm)
                    ispin = perm[idx]
                    ediff = 0.0
                    if a_field/b_field > 1
                        theta = pi * ((2.0 * rand()) - 1.0)
                    else
                        theta = a_field * pi * ((2.0 * rand()) - 1.0) / b_field
                    end
                    theta = min(max(theta + spin_vector[ispin], 0.0), pi)
                    z_mag = cos(theta) - cos(spin_vector[ispin])
                    # add z components
                    for deg = 1:maxnb
                        spin_nb = nbs_idxs[deg, ispin]
                        if spin_nb === ispin
                            ediff += b_field*nbs_vals[deg, ispin]*z_mag
                        elseif spin_nb !== 0
                            ediff += b_field*z_mag*(nbs_vals[deg, ispin]*cos(spin_vector[spin_nb]))
                        end
                    end
                    # add x components
                    ediff += a_field * (sin(spin_vector[ispin]) - sin(theta))
                    if ediff <= 0.0
                        spin_vector[ispin] = theta
                    elseif exp(-1.0 * ediff / temp) > rand()
                        spin_vector[ispin] = theta
                    end
                end
            end
        end
        # return spin_vector
    end


    function annealiq!(spin_vector::Array{Float64, 1},
                    nbs_idxs::Array{Int64, 2},
                    nbs_vals::Array{Float64, 2},
                    a_schedule::Array{Float64, 2},
                    # must be same size or will crash from @inbounds
                    b_schedule::Array{Float64, 2},
                    temp::Float64)
        maxnb = size(nbs_idxs, 1)
        nspins = size(spin_vector, 1)
        perm = collect(1:nspins)
        num_sweeps = size(a_schedule, 2)
        istep = 1
        z_mag = 0.0
        a_field = 0.0
        b_field = 0.0
        sweep = 1
        ispin = 1
        spin_nb = 1
        deg = 1
        ediff =  0.0
        theta = 0.0
        idx = 1

        @fastmath @inbounds begin
            for istep = 1:num_sweeps
                randperm!(perm)
                for idx in eachindex(perm)
                    ispin = perm[idx]
                    a_field = a_schedule[ispin, istep]
                    b_field = b_schedule[ispin, istep]
                    ediff = 0.0
                    if a_field/b_field > 1
                        theta = pi * ((2.0 * rand()) - 1.0)
                    else
                        theta = a_field * pi * ((2.0 * rand()) - 1.0) / b_field
                    end
                    theta = min(max(theta + spin_vector[ispin], 0.0), pi)
                    z_mag = cos(theta) - cos(spin_vector[ispin])
                    # add z components
                    for deg = 1:maxnb
                        spin_nb = nbs_idxs[deg, ispin]
                        if spin_nb === ispin
                            ediff += b_field*nbs_vals[deg, ispin]*z_mag
                        elseif spin_nb !== 0
                            ediff += b_field*z_mag*(nbs_vals[deg, ispin]*cos(spin_vector[spin_nb]))
                        end
                    end
                    # add x components
                    ediff += a_field * (sin(spin_vector[ispin]) - sin(theta))
                    if ediff <= 0.0
                        spin_vector[ispin] = theta
                    elseif exp(-1.0 * ediff / temp) > rand()
                        spin_vector[ispin] = theta
                    end
                end
            end
        end
        # return spin_vector
    end


    function anneals!(spin_vector::Array{Float64, 1},
                    nbs_idxs::Array{Int64, 2},
                    nbs_vals::Array{Float64, 2},
                    a_schedule::Array{Float64, 1},
                    # must be same size or will crash from @inbounds
                    b_schedule::Array{Float64, 1},
                    temp::Float64)
        # this svmc is in a sphere rather than a restricted semi-circle
        maxnb = size(nbs_idxs, 1)
        nspins = size(spin_vector, 1)
        perm = collect(1:nspins)
        phi_vector = zeros(Float64, nspins)
        istep = 1
        z_mag = 0.0
        x_mag = 0.0
        a_field = 0.0
        b_field = 0.0
        sweep = 1
        ispin = 1
        spin_nb = 1
        deg = 1
        ediff =  0.0
        theta = 0.0
        phi = 0.0
        idx = 1

        @fastmath @inbounds begin
            for istep in eachindex(a_schedule)
                a_field = a_schedule[istep]
                b_field = b_schedule[istep]
                randperm!(perm)
                for idx in eachindex(perm)
                    ispin = perm[idx]
                    ediff = 0.0
                    if a_field/b_field > 1
                        theta = pi * ((2.0 * rand()) - 1.0)
                        phi = 2 * pi * ((2.0 * rand()) - 1.0)
                    else
                        theta = a_field * pi * ((2.0 * rand()) - 1.0) / b_field
                        phi = a_field * 2 * pi * ((2.0 * rand()) - 1.0) / b_field
                    end
                    theta = min(max(theta + spin_vector[ispin], 0.0), pi)
                    phi = min(max(phi + phi_vector[ispin], -pi), pi)
                    z_mag = cos(theta) - cos(spin_vector[ispin])
                    x_mag = (sin(spin_vector[ispin])*cos(phi_vector[ispin])) - (sin(theta)*cos(phi))
                    # add z components
                    for deg = 1:maxnb
                        spin_nb = nbs_idxs[deg, ispin]
                        if spin_nb === ispin
                            ediff += b_field*nbs_vals[deg, ispin]*z_mag
                        elseif spin_nb !== 0
                            ediff += b_field*z_mag*(nbs_vals[deg, ispin]*cos(spin_vector[spin_nb]))
                        end
                    end
                    # add x components
                    ediff += a_field * x_mag
                    if ediff <= 0.0
                        spin_vector[ispin] = theta
                        phi_vector[ispin] = phi
                    elseif exp(-1.0 * ediff / temp) > rand()
                        spin_vector[ispin] = theta
                        phi_vector[ispin] = phi
                    end
                end
            end
        end
        # return spin_vector
    end


    function annealsiq!(spin_vector::Array{Float64, 1},
                    nbs_idxs::Array{Int64, 2},
                    nbs_vals::Array{Float64, 2},
                    a_schedule::Array{Float64, 2},
                    # must be same size or will crash from @inbounds
                    b_schedule::Array{Float64, 2},
                    temp::Float64)
        # this svmc is in a sphere rather than a restricted semi-circle
        maxnb = size(nbs_idxs, 1)
        nspins = size(spin_vector, 1)
        perm = collect(1:nspins)
        phi_vector = zeros(Float64, nspins)
        num_sweeps = size(a_schedule, 2)
        istep = 1
        z_mag = 0.0
        x_mag = 0.0
        a_field = 0.0
        b_field = 0.0
        sweep = 1
        ispin = 1
        spin_nb = 1
        deg = 1
        ediff =  0.0
        theta = 0.0
        phi = 0.0
        idx = 1

        @fastmath @inbounds begin
            for istep in 1:num_sweeps
                randperm!(perm)
                for idx in eachindex(perm)
                    ispin = perm[idx]
                    a_field = a_schedule[ispin, istep]
                    b_field = b_schedule[ispin, istep]
                    ediff = 0.0
                    if a_field/b_field > 1
                        theta = pi * ((2.0 * rand()) - 1.0)
                        phi = 2 * pi * ((2.0 * rand()) - 1.0)
                    else
                        theta = a_field * pi * ((2.0 * rand()) - 1.0) / b_field
                        phi = a_field * 2 * pi * ((2.0 * rand()) - 1.0) / b_field
                    end
                    theta = min(max(theta + spin_vector[ispin], 0.0), pi)
                    phi = min(max(phi + phi_vector[ispin], -pi), pi)
                    z_mag = cos(theta) - cos(spin_vector[ispin])
                    x_mag = (sin(spin_vector[ispin])*cos(phi_vector[ispin])) - (sin(theta)*cos(phi))
                    # add z components
                    for deg = 1:maxnb
                        spin_nb = nbs_idxs[deg, ispin]
                        if spin_nb === ispin
                            ediff += b_field*nbs_vals[deg, ispin]*z_mag
                        elseif spin_nb !== 0
                            ediff += b_field*z_mag*(nbs_vals[deg, ispin]*cos(spin_vector[spin_nb]))
                        end
                    end
                    # add x components
                    ediff += a_field * x_mag
                    if ediff <= 0.0
                        spin_vector[ispin] = theta
                        phi_vector[ispin] = phi
                    elseif exp(-1.0 * ediff / temp) > rand()
                        spin_vector[ispin] = theta
                        phi_vector[ispin] = phi
                    end
                end
            end
        end
        # return spin_vector
    end


    function sample(h::Dict, J::Dict; num_reads=1000, temp=0.1,
                    # for individual qubit schedules, input a
                    # dictionary of schedules
                    a_sch=1.0 .- collect(LinRange(0, 1, 2000)),
                    b_sch=collect(LinRange(0, 1, 2000)),
                    add_xtalk=false, noisy=false,
                    spherical=false, initial_state=Array{Float64, 1}())

        nspins = length(h)

        # check to see if we have individual qubit schedules
        independent_qubit_sechdules = false
        if isa(a_sch, Dict) | isa(b_sch, Dict)
            # if one input is not a dict but one if, convert to dict of lists
            if !isa(a_sch, Dict)
                schedule = deepcopy(a_sch)
                a_sch = Dict([(i, schedule) for i = 1:nspins])
            elseif !isa(b_sch, Dict)
                schedule = deepcopy(b_sch)
                b_sch = Dict([(i, schedule) for i = 1:nspins])
            end

            # check consistent inputs
            @assert length(a_sch) == nspins && length(b_sch) == nspins
            num_steps = length(a_sch[1])
            @assert all([length(a_sch[i]) == num_steps for i = 1:nspins])
            @assert all([length(b_sch[i]) == num_steps for i = 1:nspins])

            independent_qubit_sechdules = true
            a_sch_all = zeros(Float64, nspins, num_steps)
            b_sch_all = zeros(Float64, nspins, num_steps)
            for i = 1:nspins
                a_sch_all[i, :] = a_sch[i]
                b_sch_all[i, :] = b_sch[i]
            end
        end

        # prepare initial state if defined
        if length(initial_state) == nspins
            spin_vectors = [acos.(deepcopy(initial_state)) for i =1:num_reads]
        else
            # starting in the transverse field direction as default
            spin_vectors = fill(0.5 * pi, nspins, num_reads)
            spin_vectors = [spin_vectors[:, i] for i =1:num_reads]
        end
        spin_biases = Array{Float64, 1}([h[i] for i =1:nspins])
        JM, M, degs = hj_info(h, J)
        Js = copy(JM)
        nbs_idxs, nbs_vals = generate_nbr(nspins, M, maximum(degs) + 1)


        if independent_qubit_sechdules
            if spherical
                fn = annealsiq!
            else
                fn = annealiq!
            end

            if !noisy & !add_xtalk
                @inbounds for i = 1:num_reads
                    fn(spin_vectors[i], nbs_idxs, nbs_vals, a_sch_all, b_sch_all, temp)
                end
            elseif noisy & add_xtalk
                @inbounds for i = 1:num_reads
                    hc, jc = add_cross_talk(add_noise(h, J)...)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    fn(spin_vectors[i], nbs_idxs, nbs_vals, a_sch_all, b_sch_all, temp)
                end
            elseif add_xtalk & !noisy
                hc, jc = add_cross_talk(h, J)
                hj_info!(Js, M, degs, hc, jc)
                generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                @inbounds for i = 1:num_reads
                    fn(spin_vectors[i], nbs_idxs, nbs_vals, a_sch_all, b_sch_all, temp)
                end
            else
                @inbounds for i = 1:num_reads
                    hc, jc = add_noise(h, J)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    fn(spin_vectors[i], nbs_idxs, nbs_vals, a_sch_all, b_sch_all, temp)
                end
            end
        else
            if spherical
                fn = anneals!
            else
                fn = anneal!
            end
            if !noisy & !add_xtalk
                @inbounds for i = 1:num_reads
                    fn(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp)
                end
            elseif noisy & add_xtalk
                @inbounds for i = 1:num_reads
                    hc, jc = add_cross_talk(add_noise(h, J)...)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    fn(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp)
                end
            elseif add_xtalk & !noisy
                hc, jc = add_cross_talk(h, J)
                hj_info!(Js, M, degs, hc, jc)
                generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                @inbounds for i = 1:num_reads
                    fn(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp)
                end
            else
                @inbounds for i = 1:num_reads
                    hc, jc = add_noise(h, J)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    fn(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp)
                end
            end
        end
        return rotor_aggregation(spin_vectors, num_reads, JM)
    end

    export sample, anneal!, anneals!, annealiq!, annealsiq!
end
