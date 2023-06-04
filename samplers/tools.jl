module SamplingTools

    using LinearAlgebra
    using Statistics
    using Base64
    using Interpolations

    function adjacency_matrix(h::Dict, J::Dict)
        n = length(h)
        M = zeros(Float64, (n, n))
        for (i, val) in h
            M[i, i] = val
        end
        for ((x, y), val) in J
            M[x, y] = M[y, x] = val
        end
        return M
    end

    function compressed_adjacency_matrix(h::Dict, J::Dict)
        n = length(h)
        M = adjacency_matrix(h, J)
        degs = sum(M .!= 0, dims=2)
        max_deg = maximum(degs)
        compressed_M = zeros(Float64, n, max_deg)
        M_indices = zeros(Int64, n, max_deg)
        counts = ones(Int64, n)
        for w in findall(x->x!=0, M)
            i, j = w[1], w[2]
            ind = counts[i]
            compressed_M[i, ind] = M[i, j]
            M_indices[i, ind] = j
            counts[i] += 1
        end
        return compressed_M, M_indices
    end

    function hj_from_matrix(adjancy_matrix::Array{Float64, 2})
        n = size(adjancy_matrix, 1)
        h = Dict([(i, adjancy_matrix[i, i]) for i =1:n])
        J = Dict()
        for i = 1:n
            for j = 1:n
                if i < j
                    if adjancy_matrix[i, j] != 0
                        J[i, j] = adjancy_matrix[i, j]
                    end
                end
            end
        end
        return h, J
    end

    function hj_info(h::Dict, J::Dict)
        n = length(h)
        M = zeros(Float64, (n, n))
        degrees = zeros(Int64, n)
        j_full = Dict()
        for ((x, y), val) in J
            M[x, y] = M[y, x] = val
            degrees[x] += 1
            degrees[y] += 1
        end
        for (key, val) in h
            M[key, key] = val
            j_full[(key, key)] = val
        end
        return M, merge(J, j_full), degrees
    end


    function hj_info!(M::Array{Float64, 2}, j_full::Dict,
                      degrees::Array{Int64, 1}, h::Dict, J::Dict)
        fill!(degrees, 0.0)
        fill!(M, 0.0)
        j_full = Dict()
        for ((x, y), val) in J
            M[x, y] = M[y, x] = val
            degrees[x] += 1
            degrees[y] += 1
            j_full[(x, y)] = val
        end
        for (key, val) in h
            M[key, key] = val
            j_full[(key, key)] = val
        end
        return M, j_full, degrees
    end


    function generate_nbr(nspins::Int,
                          J::Dict,
                          max_degree::Int)
        # define vars
        nbs_vals = zeros(Float64, (max_degree, nspins))
        nbs_idx = zeros(Int64, (max_degree, nspins))
        ispin = 1
        ipair = 1
        x, y = 1, 1
        val = 0.0
        for ispin = 1:nspins
            ipair = 1
            for ((x, y), val) in J
                if x == ispin
                    nbs_idx[ipair, ispin] = y
                    nbs_vals[ipair, ispin] = val
                    ipair += 1
                elseif y == ispin
                    nbs_idx[ipair, ispin] = x
                    nbs_vals[ipair, ispin] = val
                    ipair += 1
                end
            end
        end
        return nbs_idx, nbs_vals
    end


    function generate_nbr!(nbs_idx::Array{Int64, 2}, nbs_vals::Array{Float64, 2},
                           nspins::Int,
                           J::Dict,
                           max_degree::Int)
        # define vars
        ispin = 1
        ipair = 1
        x, y = 1, 1
        val = 0.0
        fill!(nbs_idx, 0)
        fill!(nbs_vals, 0.0)
        for ispin = 1:nspins
            ipair = 1
            for ((x, y), val) in J
                if x == ispin
                    nbs_idx[ipair, ispin] = y
                    nbs_vals[ipair, ispin] = val
                    ipair += 1
                elseif y == ispin
                    nbs_idx[ipair, ispin] = x
                    nbs_vals[ipair, ispin] = val
                    ipair += 1
                end
            end
        end
        return nbs_idx, nbs_vals
    end


    function generate_nbr_dict(h::Dict{Int64, Float64},
                               J::Dict{Tuple{Int64, Int64}, Float64})
        # define vars
        nbs = Dict{Int64, Dict{Int64, Float64}}()
        nbs_edges = Dict()
        nspins = length(h)

        for ispin = 1:nspins
            local_nbr = Dict{Int64, Float64}(ispin => h[ispin])
            local_edges = []
            for ((x, y), val) in J
                if x == ispin
                    push!(local_nbr, y => val)
                    push!(local_edges, (x, y))
                elseif y == ispin
                    push!(local_nbr, x => val)
                    push!(local_edges, (x, y))
                end
            end
            push!(nbs, ispin => local_nbr)
            push!(nbs_edges, ispin => local_edges)
        end
        return nbs, nbs_edges
    end


    function shuffle(sorted_array::Array{Int64, 1}, n::Int64)
        t = 0
        j = 0
        @fastmath @inbounds for i = n:-1:1
            j = (rand(sorted_array) % i) + 1
            t = sorted_array[i]
            sorted_array[i] = sorted_array[j]
            sorted_array[j] = t
        end
    end


    function default_temp_range(M::Array{Float64, 2})
        JM = map(abs, M)
        min_de = minimum(diag(JM))
        max_de = maximum(sum(JM, dims=2))

        hot_temp = max_de / log(2)
        cold_temp = min_de / log(1000)
        return hot_temp, cold_temp
    end


    function ising_energy(spin_vector::Array{Float64, 1},
                          JM::Array{Float64, 2}; b_field=1.0
        )::Float64
        b = diag(JM)
        return b_field * (0.5*(spin_vector'* ((JM - diagm(b)) * spin_vector)) + (b' * spin_vector))
    end


    function discrete_aggregation(spin_vectors::Array{Array{Float64, 1}, 1},
                                  num_reads, JM::Array{Float64, 2})
        # spin_vectors = [spin_vectors[:, i] for i =1:num_reads]
        unq_vectors = unique(spin_vectors)
        num_unique = size(unq_vectors, 1)
        energies_unq = zeros(Float64, num_unique)
        occurences_unq = zeros(Int, num_unique)
        for i = 1:num_unique
            uvec = unq_vectors[i]
            energies_unq[i] = ising_energy(uvec, JM)
            occurences_unq[i] = count(x -> x == uvec, spin_vectors)
        end
        return unq_vectors, energies_unq, occurences_unq
    end


    function rotor_aggregation(rotor_vectors::Array{Array{Float64, 1}, 1}, num_reads,
                               JM::Array{Float64, 2}; cluster=false)
        if cluster
            spin_vectors = map(x-> sign.(x), map(x-> sin.(x), rotor_vectors))
        else
            spin_vectors = map(x-> sign.(x), map(x-> cos.(x), rotor_vectors))
        end
        unq_vectors = unique(spin_vectors)
        num_unique = size(unq_vectors, 1)
        energies_unq = zeros(Float64, num_unique)
        occurences_unq = zeros(Int, num_unique)
        for i = 1:num_unique
            uvec = unq_vectors[i]
            energies_unq[i] = ising_energy(uvec, JM)
            occurences_unq[i] = count(x -> x == uvec, spin_vectors)
        end
        return unq_vectors, energies_unq, occurences_unq
    end

    function pimc_aggregation(spin_lattice::Array{Array{Float64, 2}, 1},
                              num_reads, JM::Array{Float64, 2})
        # spin_vectors = [spin_vectors[:, i] for i =1:num_reads]
        nslices = size(spin_lattice[1], 2)
        spin_vectors = Array{Array{Float64, 1}, 1}()
        for i = 1:num_reads
            v, e, o = discrete_aggregation([spin_lattice[i][:, j] for j =1:nslices], nslices, JM)
            push!(spin_vectors, collect(v[findmin(e)[2]]))
        end
        return discrete_aggregation(spin_vectors, num_reads, JM)
    end


    function bootstrap(sample_set::Array{Float64, 1},
                       bootstraps=1000, return_bounds=true)
        # for finding a bootstrap median
        n = length(sample_set)
        sample_set = repeat(sample_set, outer=(1,bootstraps))
        @inbounds bootstrap_samples = sample_set[rand(1:n, n, bootstraps)]
        medians = vcat(median(bootstrap_samples, dims=1)...)
        m = median(medians)
        if return_bounds
            bounds = quantile(medians, [0.05, 0.95])
            return m, bounds
        end
        return m
    end


    function add_cross_talk(h::Dict, j::Dict, susceptibility=-0.035)
        nbs, nbs_edges = generate_nbr_dict(h, j)
        j_p = merge(j, Dict([((y, x), val) for ((x, y), val) in j]))
        h_c = copy(h)
        j_c = copy(j)
        j_p_keys = collect(keys(j_p))
        j_c_keys = collect(keys(j_c))
        for (node, n_edges) in nbs_edges
            h_off = 0.0
            ne = length(n_edges)
            for i1 = 1:ne
                (x, y) = n_edges[i1]
                if node == y
                    k = x
                else
                    k = y
                end
                h_off += j[(x, y)] * h[k]
                for i2 = 1:ne
                    (u, v) = n_edges[i2]
                    if ((x, y) != (u, v)) & (i1 < i2)
                        if v == node
                            a = u
                        else
                            a = v
                        end
                        if k < a
                            jij = (k, a)
                        else
                            jij = (a, k)
                        end
                        j_off = susceptibility * get(j_p, (node, a), 0.0) * get(j_p, (node, k), 0.0)
                        if (jij in j_c_keys) == false
                            push!(j_c_keys, jij)
                            push!(j_c, jij => 0.0)
                        end
                        j_c[jij] += j_off
                    end
                end
            end
            h_c[node] += susceptibility * h_off
        end
        return h_c, j_c
    end


    function add_noise(h, j, h_std=0.03, j_std=0.03, h_mean=0.0, j_mean=0.0)
        h_noise = (h_std .* randn(length(h))) .+ h_mean
        j_noise = (j_std .* randn(length(j))) .+ j_mean
        h_new = Dict(Tuple(zip(keys(h), values(h) .+ h_noise)))
        j_new = Dict(Tuple(zip(keys(j), values(j) .+ j_noise)))
        return h_new, j_new
    end


    function PFC_dict(d::Float64; N=2, R=1.0)
        h = Dict{Int64, Float64}([(i, 0.0) for i = 1:(2*N)])
        j = Dict{Tuple, Float64}()
        @assert R > 0.0
        for i = 1:N
            h[i] = R * (1.0 - d)
            h[i+N] = -1.0 * R
            j[(i, i+N)] = -1.0 * R
            if i < N
                j[(i, i+1)] = -1.0 * R
            end
        end
        return h, j
    end


    function energy_ranges(h::Dict{Int64, Float64}, j::Dict{Tuple, Float64},
                           max_field=1.0)
        n = length(h)
        @assert n >= 1
        abs_h = []
        abs_dict = Dict{Int64, Float64}()
        abs_j = []
        # bias contrib
        for (key, hh) in h
            abs_h_dict[key] = abs(hh)
            if hh != 0.0
                push!(abs_h, abs(hh))
            end
        end
        # j contrib
        for ((i1, i2), jj) in j
            ajj = abs(jj)
            abs_dict[i1] += ajj
            abs_dict[i2] += ajj
            if jj != 0.0
                push!(abs_j, ajj)
            end
        end

        abs_vals = vcat(abs_h, abs_j)
        if length(abs_vals) == 0
            return [0.0, 0.0]
        end

        min_de = minimum(abs_vals)
        max_de = maximum(values(abs_dict))
        return [min_de, max_de]
    end


    function DQA_schedule(k::Int, nspins::Int, a_schedule::Array{Float64, 1},
                          b_schedule::Array{Float64, 1}; sx=0.2, cx=0.0, c1=0.0)
        num_steps = length(a_schedule)
        new_steps = convert(Int64, ceil(num_steps / (1.0 - sx)))
        s_old = LinRange(sx, 1.0, num_steps)
        s = collect(LinRange(0.0, 1.0, new_steps))
        ak_sch = collect(LinRange(cx, c1, num_steps))
        a_dqa_dict = Dict()
        b_dqa_dict = Dict()
        for i = 1:nspins
            a_sch = deepcopy(a_schedule)
            b_sch = deepcopy(b_schedule)
            if i == k
                afn = monotonic_interpolation(s_old, ak_sch)
                a_dqa_dict[i] = afn.(s)
                b_dqa_dict[i] = LinRange(-sx/(1.0-sx), 1.0, new_steps)
            else
                afn = monotonic_interpolation(s_old, a_sch)
                bfn = monotonic_interpolation(s_old, b_sch)
                a_dqa_dict[i] = afn.(s)
                b_dqa_dict[i] = bfn.(s)
            end
        end
        return a_dqa_dict, b_dqa_dict
    end


    function monotonic_interpolation(x, y)
        return extrapolate(interpolate(x, y, SteffenMonotonicInterpolation()), Flat())
    end

    function TTS(probability::Number, time::Number; pd=0.99, max_time=1.0e9)
        p = min(probability, pd)
        if p == 0.0
            return max_time
        end
        return time * (log(1.0 - pd)/ log(1.0 - p))
    end

    function pause_schedule(_fn, sp, τ)
        # τ is the fraction of the total anneal for which we pause
        # _fn has to have domain [0, 1]
        ta = 1.0 - τ
        sp2 = sp*ta
        sp3 = sp2 + τ
        function p_fn(s)
            if s <= sp2
                return _fn(s/ta)
            elseif sp2 < s <= sp3
                return _fn(sp)
            else
                return _fn((s-τ)/ta)
            end
        end
        return p_fn
    end

    function make_monte_carlo_schedule(s_schedule, time_schedule,
                                       a_fn, b_fn, steps_per_mus)
        sch = vcat([collect(LinRange(s_schedule[i], s_schedule[i+1],
                                     convert(Int, round(time_schedule[i] *
                                     steps_per_mus)))) for i = 1:length(time_schedule)]...)
        return a_fn.(sch), b_fn.(sch)
    end

    export rotor_aggregation, discrete_aggregation, ising_energy,
            generate_nbr, default_temp_range, hj_info, shuffle,
            generate_nbr_dict, bootstrap, add_cross_talk, add_noise,
            hj_info!, generate_nbr!, pimc_aggregation, PFC_dict,
            energy_ranges, DQA_schedule, monotonic_interpolation,
            adjacency_matrix, hj_from_matrix, TTS, pause_schedule,
            make_monte_carlo_schedule
end
