"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com

Based off the work published in
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.66.094203
Quantum annealing by the path-integral Monte Carlo method:
The two-dimensional random Ising model
Roman Martoňák, Giuseppe E. Santoro, and Erio Tosatti
Phys. Rev. B 66, 094203 – Published 13 September 2002
There is also a method that allows for more complex schedules and one that
implements a bosonic bath from work published in
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.94.047201
Phase Diagram and Critical Exponents of a Dissipative Ising Spin Chain in a
Transverse Magnetic Field
Philipp Werner, Klaus Völker, Matthias Troyer, and Sudip Chakravarty
Phys. Rev. Lett. 94, 047201 – Published 31 January 2005
"

module PIMC
    include("tools.jl")
    include("SA.jl")
    using ..SA
    using ..SamplingTools: hj_info, default_temp_range, generate_nbr_dict,
                            generate_nbr, pimc_aggregation, add_noise,
                            add_cross_talk, hj_info!, generate_nbr!
    using Random
    using Distributed
    using LinearAlgebra
    using SharedArrays
    function anneal!(spin_lattice::Array{Float64, 2},
                    nbs_idxs::Array{Int64, 2},
                    nbs_vals::Array{Float64, 2},
                    a_schedule::Array{Float64, 1},
                    # must be same size or will crash from @inbounds
                    b_schedule::Array{Float64, 1},
                    temp::Float64,
                    local_flips::Bool)
        maxnb = size(nbs_idxs, 1)
        nspins = size(spin_lattice, 1)
        nslices = size(spin_lattice, 2)
        perm = collect(1:nspins)
        istep = 1
        a_field = 0.0
        b_field = 0.0
        # the separation of states in imaginary time
        τ = 1 / (Float64(nslices) * temp)
        jperp = 0.0
        islice = 0
        idx = 1
        ispin = 1
        spin_nb = 1
        deg = 1
        ediff_prob =  0.0
        ediff_trans =  0.0
        if !local_flips
            @fastmath @inbounds begin
                for istep in eachindex(a_schedule)
                    a_field = a_schedule[istep]
                    b_field = -2.0 * τ * b_schedule[istep]
                    jperp = -log(tanh(τ * a_field))
                    # perform local moves
                    for islice = 1:nslices
                        randperm!(perm)
                        for idx in eachindex(perm)
                            ispin = perm[idx]
                            ediff = 0.0
                            # add z components
                            for deg = 1:maxnb
                                spin_nb = nbs_idxs[deg, ispin]
                                if spin_nb === ispin
                                    ediff += b_field*nbs_vals[deg, ispin]*spin_lattice[ispin, islice]
                                elseif spin_nb !== 0
                                    ediff += b_field*spin_lattice[ispin, islice]*(nbs_vals[deg, ispin]*spin_lattice[spin_nb, islice])
                                end
                            end
                            # add x components
                            # periodic boundaries
                            if islice == 1
                                tleft = nslices
                                tright = 2
                            elseif islice == nslices
                                tleft = nslices-1
                                tright = 1
                            else
                                tleft = islice-1
                                tright = islice+1
                            end
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tleft])
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tright])
                            if ediff <= 0.0
                                spin_lattice[ispin, islice] *= -1.0
                            elseif exp(-ediff) > rand()
                                spin_lattice[ispin, islice] *= -1.0
                            end
                        end
                    end
                    # perform global moves
                    randperm!(perm)
                    for idx in eachindex(perm)
                        ispin = perm[idx]
                        ediff = 0.0
                        for islice = 1:nslices
                            for deg = 1:maxnb
                                spin_nb = nbs_idxs[deg, ispin]
                                if spin_nb === ispin
                                    ediff += b_field*nbs_vals[deg, ispin]*spin_lattice[ispin, islice]
                                elseif spin_nb !== 0
                                    ediff += b_field*spin_lattice[ispin, islice]*(nbs_vals[deg, ispin]*spin_lattice[spin_nb, islice])
                                end
                            end
                        end
                        if ediff <= 0.0
                            spin_lattice[ispin, :] *= -1.0
                        elseif exp(-ediff) > rand()
                            spin_lattice[ispin, :] *= -1.0
                        end
                    end
                end
            end
        else
            @fastmath @inbounds begin
                for istep in eachindex(a_schedule)
                    a_field = a_schedule[istep]
                    b_field = -2.0 * τ * b_schedule[istep]
                    jperp = -log(tanh(τ * a_field))
                    # perform local moves
                    for islice = 1:nslices
                        randperm!(perm)
                        for idx in eachindex(perm)
                            ispin = perm[idx]
                            ediff = 0.0
                            # add z components
                            for deg = 1:maxnb
                                spin_nb = nbs_idxs[deg, ispin]
                                if spin_nb === ispin
                                    ediff += b_field*nbs_vals[deg, ispin]*spin_lattice[ispin, islice]
                                elseif spin_nb !== 0
                                    ediff += b_field*spin_lattice[ispin, islice]*(nbs_vals[deg, ispin]*spin_lattice[spin_nb, islice])
                                end
                            end
                            # add x components
                            # periodic boundaries
                            if islice == 1
                                tleft = nslices
                                tright = 2
                            elseif islice == nslices
                                tleft = nslices-1
                                tright = 1
                            else
                                tleft = islice-1
                                tright = islice+1
                            end
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tleft])
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tright])
                            if ediff <= 0.0
                                spin_lattice[ispin, islice] *= -1.0
                            elseif exp(-ediff) > rand()
                                spin_lattice[ispin, islice] *= -1.0
                            end
                        end
                    end
                end
            end
        end
        # return spin_vector
    end

    function annealiq!(spin_lattice::Array{Float64, 2},
                    nbs_idxs::Array{Int64, 2},
                    nbs_vals::Array{Float64, 2},
                    a_schedule::Array{Float64, 2},
                    # must be same size or will crash from @inbounds
                    b_schedule::Array{Float64, 2},
                    temp::Float64,
                    local_flips::Bool)
        maxnb = size(nbs_idxs, 1)
        nspins = size(spin_lattice, 1)
        nslices = size(spin_lattice, 2)
        perm = collect(1:nspins)
        num_sweeps = size(a_schedule, 2)
        istep = 1
        a_field = 0.0
        b_field = 0.0
        τ = 1 / (Float64(nslices) * temp)
        jperp = 0.0
        islice = 0
        idx = 1
        ispin = 1
        spin_nb = 1
        deg = 1
        ediff =  0.0

        if !local_flips
            @fastmath @inbounds begin
                for istep = 1:num_sweeps
                    # perform local moves
                    for islice = 1:nslices
                        randperm!(perm)
                        for idx in eachindex(perm)
                            ispin = perm[idx]
                            a_field = a_schedule[ispin, istep]
                            b_field = -2.0 * τ * b_schedule[ispin, istep]
                            jperp = -log(tanh(τ * a_field))
                            ediff = 0.0
                            # add z components
                            for deg = 1:maxnb
                                spin_nb = nbs_idxs[deg, ispin]
                                if spin_nb === ispin
                                    ediff += b_field*nbs_vals[deg, ispin]*spin_lattice[ispin, islice]
                                elseif spin_nb !== 0
                                    ediff += b_field*spin_lattice[ispin, islice]*(nbs_vals[deg, ispin]*spin_lattice[spin_nb, islice])
                                end
                            end
                            # add x components
                            # periodic boundaries
                            if islice == 1
                                tleft = nslices
                                tright = 2
                            elseif islice == nslices
                                tleft = nslices-1
                                tright = 1
                            else
                                tleft = islice-1
                                tright = islice+1
                            end
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tleft])
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tright])
                            if ediff <= 0.0
                                spin_lattice[ispin, islice] *= -1.0
                            elseif exp(-ediff) > rand()
                                spin_lattice[ispin, islice] *= -1.0
                            end
                        end
                    end
                    # perform global moves
                    randperm!(perm)
                    for idx in eachindex(perm)
                        ispin = perm[idx]
                        b_field = b_schedule[ispin, istep]
                        ediff = 0.0
                        for islice = 1:nslices
                            for deg = 1:maxnb
                                spin_nb = nbs_idxs[deg, ispin]
                                if spin_nb === ispin
                                    ediff += b_field*nbs_vals[deg, ispin]*spin_lattice[ispin, islice]
                                elseif spin_nb !== 0
                                    ediff += b_field*spin_lattice[ispin, islice]*(nbs_vals[deg, ispin]*spin_lattice[spin_nb, islice])
                                end
                            end
                        end
                        if ediff <= 0.0
                            spin_lattice[ispin, :] *= -1.0
                        elseif exp(-ediff) > rand()
                            spin_lattice[ispin, :] *= -1.0
                        end
                    end
                end
            end
        else
            @fastmath @inbounds begin
                for istep = 1:num_sweeps
                    # perform local moves
                    for islice = 1:nslices
                        randperm!(perm)
                        for idx in eachindex(perm)
                            ispin = perm[idx]
                            a_field = a_schedule[ispin, istep]
                            b_field = -2.0 * τ * b_schedule[ispin, istep]
                            jperp = -log(tanh(τ * a_field))
                            ediff = 0.0
                            # add z components
                            for deg = 1:maxnb
                                spin_nb = nbs_idxs[deg, ispin]
                                if spin_nb === ispin
                                    ediff += b_field*nbs_vals[deg, ispin]*spin_lattice[ispin, islice]
                                elseif spin_nb !== 0
                                    ediff += b_field*spin_lattice[ispin, islice]*(nbs_vals[deg, ispin]*spin_lattice[spin_nb, islice])
                                end
                            end
                            # add x components
                            # periodic boundaries
                            if islice == 1
                                tleft = nslices
                                tright = 2
                            elseif islice == nslices
                                tleft = nslices-1
                                tright = 1
                            else
                                tleft = islice-1
                                tright = islice+1
                            end
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tleft])
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tright])
                            if ediff <= 0.0
                                spin_lattice[ispin, islice] *= -1.0
                            elseif exp(-ediff) > rand()
                                spin_lattice[ispin, islice] *= -1.0
                            end
                        end
                    end
                end
            end
        end
        # return spin_vector
    end

    function annealb!(spin_lattice::Array{Float64, 2},
                    nbs_idxs::Array{Int64, 2},
                    nbs_vals::Array{Float64, 2},
                    a_schedule::Array{Float64, 1},
                    # must be same size or will crash from @inbounds
                    b_schedule::Array{Float64, 1},
                    temp::Float64,
                    α::Float64,
                    local_flips::Bool)
        maxnb = size(nbs_idxs, 1)
        nspins = size(spin_lattice, 1)
        nslices = size(spin_lattice, 2)
        perm = collect(1:nspins)
        istep = 1
        a_field = 0.0
        b_field = 0.0
        # the separation of states in imaginary time
        τ = 1 / (Float64(nslices) * temp)
        jperp = 0.0
        islice = 0
        idx = 1
        ispin = 1
        spin_nb = 1
        deg = 1
        ediff =  0.0
        cslice = 0
        cind = 0
        distance_factor = (pi/Float64(nslices))^2 .* α ./ (sin.((pi .* collect(1:(nslices-1))) ./ Float64(nslices))).^2
        if !local_flips
            @fastmath @inbounds begin
                for istep in eachindex(a_schedule)
                    a_field = a_schedule[istep]
                    b_field = -2.0 * τ * b_schedule[istep]
                    jperp = -log(tanh(τ * a_field))
                    # perform local moves
                    for islice = 1:nslices
                        randperm!(perm)
                        for idx in eachindex(perm)
                            ispin = perm[idx]
                            ediff = 0.0
                            # add z components
                            for deg = 1:maxnb
                                spin_nb = nbs_idxs[deg, ispin]
                                if spin_nb === ispin
                                    ediff += b_field*nbs_vals[deg, ispin]*spin_lattice[ispin, islice]
                                elseif spin_nb !== 0
                                    ediff += b_field*spin_lattice[ispin, islice]*(nbs_vals[deg, ispin]*spin_lattice[spin_nb, islice])
                                end
                            end
                            # add x components
                            # periodic boundaries
                            if islice == 1
                                tleft = nslices
                                tright = 2
                            elseif islice == nslices
                                tleft = nslices-1
                                tright = 1
                            else
                                tleft = islice-1
                                tright = islice+1
                            end
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tleft])
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tright])
                            for cslice = 1:(nslices-1)
                                cind = (((cslice-1) + islice) % nslices) + 1
                                # note this is not whats in the paper but an equiv.
                                ediff += distance_factor[abs(islice - cind)] * spin_lattice[ispin, islice] * spin_lattice[ispin, cind]
                            end

                            if ediff <= 0.0
                                spin_lattice[ispin, islice] *= -1.0
                            elseif exp(-ediff) > rand()
                                spin_lattice[ispin, islice] *= -1.0
                            end
                        end
                    end
                    # perform global moves
                    randperm!(perm)
                    for idx in eachindex(perm)
                        ispin = perm[idx]
                        ediff = 0.0
                        for islice = 1:nslices
                            for deg = 1:maxnb
                                spin_nb = nbs_idxs[deg, ispin]
                                if spin_nb === ispin
                                    ediff += b_field*nbs_vals[deg, ispin]*spin_lattice[ispin, islice]
                                elseif spin_nb !== 0
                                    ediff += b_field*spin_lattice[ispin, islice]*(nbs_vals[deg, ispin]*spin_lattice[spin_nb, islice])
                                end
                            end
                        end
                        if ediff <= 0.0
                            spin_lattice[ispin, :] *= -1.0
                        elseif exp(-ediff) > rand()
                            spin_lattice[ispin, :] *= -1.0
                        end
                    end
                end
            end
        else
            @fastmath @inbounds begin
                for istep in eachindex(a_schedule)
                    a_field = a_schedule[istep]
                    b_field = -2.0 * τ * b_schedule[istep]
                    jperp = -log(tanh(τ * a_field))
                    # perform local moves
                    for islice = 1:nslices
                        randperm!(perm)
                        for idx in eachindex(perm)
                            ispin = perm[idx]
                            ediff = 0.0
                            # add z components
                            for deg = 1:maxnb
                                spin_nb = nbs_idxs[deg, ispin]
                                if spin_nb === ispin
                                    ediff += b_field*nbs_vals[deg, ispin]*spin_lattice[ispin, islice]
                                elseif spin_nb !== 0
                                    ediff += b_field*spin_lattice[ispin, islice]*(nbs_vals[deg, ispin]*spin_lattice[spin_nb, islice])
                                end
                            end
                            # add x components
                            # periodic boundaries
                            if islice == 1
                                tleft = nslices
                                tright = 2
                            elseif islice == nslices
                                tleft = nslices-1
                                tright = 1
                            else
                                tleft = islice-1
                                tright = islice+1
                            end
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tleft])
                            ediff += spin_lattice[ispin, islice]*(jperp*spin_lattice[ispin, tright])
                            for cslice = 1:(nslices-1)
                                cind = (((cslice-1) + islice) % nslices) + 1
                                # note this is not whats in the paper but an equiv.
                                ediff += distance_factor[abs(islice - cind)] * spin_lattice[ispin, islice] * spin_lattice[ispin, cind]
                            end

                            if ediff <= 0.0
                                spin_lattice[ispin, islice] *= -1.0
                            elseif exp(-ediff) > rand()
                                spin_lattice[ispin, islice] *= -1.0
                            end
                        end
                    end
                end
            end
        end
        # return spin_vector
    end


    function sample(h::Dict, J::Dict; num_reads=10000, temp=0.1,
                    a_sch=1.0 .- collect(LinRange(0, 1, 1000)),
                    b_sch=collect(LinRange(0, 1, 1000)), nslices=64,
                    add_xtalk=false, noisy=false, pre_anneal=false,
                    add_bath=false, coupling_const=0.001, local_flips=true,
                    initial_state=Array{Float64, 1}(), return_raw=false)
        # forward anneal usage only
        independent_qubit_sechdules = false
        nspins = length(h)
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

        if independent_qubit_sechdules & add_bath
            @error "Bath and individual schedules not yet implemeneted"
        end

        if length(initial_state) == length(h)
            spin_array = zeros(Float64, nspins, nslices)
            for islice = 1:nslices
                spin_array[:, islice] = deepcopy(initial_state)
            end
            spin_vectors = [deepcopy(spin_array) for i =1:num_reads]
        else
            if pre_anneal
                result = SA.sample(h, J, num_reads=nslices*num_reads, num_steps=length(a_sch), return_raw=true)
                spin_vectors = [Array{Float64, 2}(hcat(result[1+(i-1)*nslices:i*nslices]...)) for i =1:num_reads]
            else
                spin_vectors = rand([-1.0, 1.0], nspins, nslices, num_reads)
                spin_vectors = [spin_vectors[:, :, i] for i =1:num_reads]
            end
        end
        JM, M, degs = hj_info(h, J)
        Js = copy(JM)
        nbs_idxs, nbs_vals = generate_nbr(nspins, M, maximum(degs) + 1)
        if add_bath
            if !noisy & !add_xtalk
                @inbounds for i = 1:num_reads
                    annealb!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp, coupling_const, local_flips)
                end
            elseif noisy & add_xtalk
                @inbounds for i = 1:num_reads
                    hc, jc = add_cross_talk(add_noise(h, J)...)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    annealb!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp, coupling_const, local_flips)
                end
            elseif add_xtalk & !noisy
                hc, jc = add_cross_talk(h, J)
                hj_info!(Js, M, degs, hc, jc)
                generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                @inbounds for i = 1:num_reads
                    annealb!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp, coupling_const, local_flips)
                end
            else
                @inbounds for i = 1:num_reads
                    hc, jc = add_noise(h, J)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    annealb!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp, coupling_const, local_flips)
                end
            end
        elseif independent_qubit_sechdules
            if !noisy & !add_xtalk
                @inbounds for i = 1:num_reads
                    annealiq!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch_all, b_sch_all, temp, local_flips)
                end
            elseif noisy & add_xtalk
                @inbounds for i = 1:num_reads
                    hc, jc = add_cross_talk(add_noise(h, J)...)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    annealiq!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch_all, b_sch_all, temp, local_flips)
                end
            elseif add_xtalk & !noisy
                hc, jc = add_cross_talk(h, J)
                hj_info!(Js, M, degs, hc, jc)
                generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                @inbounds for i = 1:num_reads
                    annealiq!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch_all, b_sch_all, temp, local_flips)
                end
            else
                @inbounds for i = 1:num_reads
                    hc, jc = add_noise(h, J)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    annealiq!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch_all, b_sch_all, temp, local_flips)
                end
            end
        else
            if !noisy & !add_xtalk
                @inbounds for i = 1:num_reads
                    anneal!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp, local_flips)
                end
            elseif noisy & add_xtalk
                @inbounds for i = 1:num_reads
                    hc, jc = add_cross_talk(add_noise(h, J)...)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    anneal!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp, local_flips)
                end
            elseif add_xtalk & !noisy
                hc, jc = add_cross_talk(h, J)
                hj_info!(Js, M, degs, hc, jc)
                generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                @inbounds for i = 1:num_reads
                    anneal!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp, local_flips)
                end
            else
                @inbounds for i = 1:num_reads
                    hc, jc = add_noise(h, J)
                    hj_info!(Js, M, degs, hc, jc)
                    generate_nbr!(nbs_idxs, nbs_vals, nspins, M, maximum(degs) + 1)
                    anneal!(spin_vectors[i], nbs_idxs, nbs_vals, a_sch, b_sch, temp, local_flips)
                end
            end
        end
        if return_raw
            return spin_vectors
        else
            return pimc_aggregation(spin_vectors, num_reads, JM)
        end
    end

    export sample, anneal!, annealiq!, annealb!
end
