"
Author: Daniel O'Connor
Email: dt.oconnor@outlook.com

"

module VonNeumann
    include("..\\utils\\sampling_utils.jl")
    include("..\\utils\\analytical_utils.jl")
    using OpenQuantumTools, OrdinaryDiffEq, Plots
    using QuadGK
    using NPZ
    using Arpack
    using Interpolations
    using ..QUtils: construct_H
    using ..SamplingUtils: add_noise, add_cross_talk, ising_energy

    function solve(hz::Dict, jz::Dict, tf::Float64, afn, bfn; levels=2,
                   cross_talk=false, return_raw=false, magnetization=false,
                   noise=false, saveat=nothing, use_sparse=false,
                   lstf_dqa=false, k=1, cafn=nothing, cbfn=nothing,
                   bias_state=nothing, initial_state=nothing, param_fn=nothing,
                   abstol=1e-10, reltol=1e-9, use_dm=true)
        N = length(hz)

        afnc = (s) -> complex(afn(s))
        bfnc = (s) -> complex(bfn(s))
        dqa = false
        if lstf_dqa && (cafn != nothing && cbfn != nothing)
            dqa = true
            cafnc = (s) -> complex(cafn(s))
            cbfnc = (s) -> complex(cbfn(s))
        end

        hx = Dict([(i, -1.0) for i =1:N])
        jx = Dict()
        if noise
            hz, jz = add_noise(hz, jz)
        end
        if cross_talk
            hz, jz = add_cross_talk(hz, jz)
        end
        basis = PauliVec[1][1]
        if use_sparse
            Hz, sub_spaces = construct_H(hz, jz, spσz, return_subspaces=true)
            Hx, xsub_spaces = construct_H(hx, jx, spσx, return_subspaces=true)
            # basis = sparse(basis)
            Hfn = SparseHamiltonian
        else
            Hz, sub_spaces = construct_H(hz, jz, σz, return_subspaces=true)
            Hx, xsub_spaces = construct_H(hx, jx, σx, return_subspaces=true)
            Hfn = DenseHamiltonian
        end
        if dqa
            if bias_state == nothing
                b2 = hz[k] >= 0 ? PauliVec[3][1] : PauliVec[3][2]
            else
                b2 = bias_state >= 0 ? PauliVec[3][1] : PauliVec[3][2]
            end
            Hqz = hz[k] .* sub_spaces[k]
            Hz_dqa = Hz - Hqz
            Hqx = hx[k] .* xsub_spaces[k]
            Hx_dqa = Hx - Hqx
            H = Hfn([afnc, bfnc, cafnc, cbfnc], [Hx_dqa, Hz_dqa, Hqx, Hqz])
            u0 = kron([i==k ? b2 : basis for i =1:N]...)
        else
            H = Hfn([afnc, bfnc], [Hx, Hz])
            u0 = kron([basis for i =1:N]...)
        end
        if initial_state != nothing
            @assert size(initial_state, 1) == 2^N
            u0 = initial_state
        end
        if use_dm
            u0 = u0 * u0'
        end
        if param_fn == nothing
            annealing_parameter = (tf, t) -> t / tf
        else
            annealing_parameter = param_fn
        end
        annealing = Annealing(H, u0, annealing_parameter=annealing_parameter)
        if saveat === nothing
            saveat = 0:0.01:tf
        end
        
        is_dm = length(size(u0)) == 2
        if is_dm
            sols = solve_von_neumann(annealing, tf, alg=Tsit5(), saveat=saveat,
                                    abstol=abstol, reltol=reltol)
        else
            sols = solve_schrodinger(annealing, tf, alg=Tsit5(), saveat=saveat,
                                     abstol=abstol, reltol=reltol)
        end

        if return_raw
            return sols
        end
        p = zeros(Float64, levels, length(saveat))
        for i = 1:length(saveat)
            val, states = eigen_decomp(H, saveat[i]/tf, lvl=levels)
            # _, states = eigs(H(saveat[i]/tf), nev=levels)
            for j = 1:levels
                if is_dm
                    p[j, i] = real(states[:, j]' * (sols[i] * states[:, j]))
                else
                    p[j, i] = real(abs(sols[i]' * states[:, j])^2)
                end
            end
        end
        if magnetization
            mags = zeros(Float64, N, length(saveat))
            for i = 1:length(saveat)
                for j = 1:N
                    if is_dm
                        mags[j, i] = real(tr(sub_spaces[j] * sols[i]))
                    else
                        mags[j, i] = real(tr(sub_spaces[j] * (sols[i] * sols[i]')))
                    end
                end
            end
            return p, mags, sols
        end
        return p, sols
    end

    export solve
end
