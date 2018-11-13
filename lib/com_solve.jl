include("../lib/quantum_utils.jl")
using DifferentialEquations

#######################################################################################
#
# define Schrodinger equation problem with dispalcement operator
#   T: final time
#   ψ0_qo: initial ket  (currently only vaccum state is allowed)
#   A,B1,B2,C,D: tensors for Hamiltonian  (B= )
#
#
function Schrodinger_com(T::Float64,ψ0::Ket,μ₁0::Array{Float64},A::Array{Float64},B1::Array{Float64},B2::Array{Float64},C::Array{Float64},D::Array{Float64})

    Nfocks = ψ0.basis.shape .-1
    N = length(Nfocks)
    as = a_vectors(Nfocks)
    adags = dagger.(as);
    z = vcat((as+adags),(as-adags)/im)/√2;

    Ω_uu = Omega_uu(N)
    Ω_dd = Omega_dd(N)
    μ₁ = zeros(Float64,2N)
    dμ₁ = zeros(Float64,2N)
    ϕ = ground_state(Nfocks)
    ψ = ground_state(Nfocks)

    state_vec0 = vcat(μ₁0,real.(ψ0.data),imag.(ψ0.data))

    function time_evol_det(d_state_vec,state_vec,p,t)

        μ₁ = state_vec[1:2N]
        ϕ.data = state_vec[2N+1:2N+length(ϕ.data)] + im * state_vec[2N+1+length(ϕ.data):2N+length(ϕ.data)*2]

        B = B1 + B2 * t/T

        μ₂ = get_μ(ϕ,z,2)
        μ₃ = get_μ(ϕ,z,3)
        μ₂_raw = get_2nd_raw_moment(μ₁,μ₂)
        μ₃_raw = get_3rd_raw_moment(μ₁,μ₂,μ₃)


        @tensor begin
            dμ₁[i]       = -1. * A[j]       * Ω_dd[j,i]
            dμ₁[i]      += -2. * B[j,k]     * Ω_dd[j,i] * μ₁[k] #
            dμ₁[i]      += -3. * C[j,k,l]   * Ω_dd[j,i] * μ₂_raw[k,l]
            dμ₁[i]      += -4. * D[j,k,l,m] * Ω_dd[j,i] * μ₃_raw[k,l,m]
        end

        @tensor f_dis[k] := -Ω_uu[i,k]*dμ₁[i]
        F_dis = get_operator(f_dis,z)

        z_prime = z + μ₁ .* [identityoperator(Nfocks) for i in 1:2N]
        Ht = (get_operator(A,z_prime) + get_operator(B,z_prime) + get_operator(C,z_prime) + get_operator(D,z_prime)) - F_dis

        dϕ = -im*Ht * ϕ

        d_state_vec .= vcat(dμ₁,real.(dϕ.data),imag.(dϕ.data))
    end

    return ODEProblem(time_evol_det,state_vec0,(0.0,T))

end



#######################################################################################
#
# define Belavkin equation problem with dispalcement operator
#   T: final time
#   ψ0_qo: initial ket  (currently only vaccum state is allowed)
#   A,B1,B2,C,D: tensors for Hamiltonian  (B= )
#
#
function Belavkin_com(T::Float64,ψ0::Ket,μ₁0::Array{Float64},A::Array{Float64},B1::Array{Float64},B2::Array{Float64},C::Array{Float64},D::Array{Float64},vec_EF::Array{Array{Complex{Float64}},2};seed_input=0)

    Nfocks = ψ0.basis.shape .-1
    N = length(Nfocks)
    as = a_vectors(Nfocks)
    adags = dagger.(as);
    z = vcat((as+adags),(as-adags)/im)/√2;

    Ω_uu = Omega_uu(N)
    Ω_dd = Omega_dd(N)
    μ₁ = zeros(Float64,2N)
    dμ₁ = zeros(Float64,2N)
    ϕ = ground_state(Nfocks)
    dϕ = ground_state(Nfocks)
    ψ = ground_state(Nfocks)

    A′= Array{Complex{Float64}}(undef,2N);
    B′= Array{Complex{Float64}}(undef,2N,2N);
    C′= Array{Complex{Float64}}(undef,2N,2N,2N);
    D′= Array{Complex{Float64}}(undef,2N,2N,2N,2N);


    state_vec0 = vcat(μ₁0,real.(ψ0.data),imag.(ψ0.data))

    function time_evol_det(d_state_vec,state_vec,p,t)

        μ₁ = state_vec[1:2N]
        ϕ.data = state_vec[2N+1:2N+length(ϕ.data)] + im * state_vec[2N+1+length(ϕ.data):2N+length(ϕ.data)*2]

        B = B1 + B2 * t/T

        μ₂ = get_μ(ϕ,z,2)
        μ₂_raw = get_2nd_raw_moment(μ₁,μ₂)
        get_effective_Hamiltonian_tensors!(A′,B′,C′,D′,A,B,C,D,vec_EF,μ₁,μ₂_raw)
        z_prime = z + μ₁ .* [identityoperator(Nfocks) for i in 1:2N]
        H = get_operator([A′,B′,C′,D′],z_prime)

        mul!(dϕ.data,-im*H.data,ϕ.data)
        dif_norm = real(dagger(ϕ) * dϕ)
        dϕ.data -= dif_norm*ϕ.data

        dμ₁ = [2*real(dagger(ϕ) * z_prime_i * dϕ) for z_prime_i in z_prime ]
        @tensor f_dis[k] := -Ω_uu[i,k]*dμ₁[i]
        dϕ += im * get_operator(f_dis,z) * ϕ # -F_dis

        d_state_vec .= vcat(dμ₁,real.(dϕ.data),imag.(dϕ.data))
    end

    #########################################################################
    #
    # stochastic part
    #
    function time_evol_sto(d_state_vec,state_vec,p,t)

        μ₁ = state_vec[1:2N]
        ϕ.data = state_vec[2N+1:2N+length(ϕ.data)] + im * state_vec[2N+1+length(ϕ.data):2N+length(ϕ.data)*2]


        for i in 1:size(vec_EF)[1]

            z_prime = z + μ₁ .* [identityoperator(Nfocks) for i in 1:2N]
            L = get_operator(vec_EF[i,1],z_prime) + get_operator(vec_EF[i,2],z_prime)


            mul!(dϕ.data,L.data,ϕ.data)
            dif_norm = real(dagger(ϕ) * dϕ)
            dϕ.data -= dif_norm*ϕ.data

            dμ₁ = [2*real(dagger(ϕ) * z_prime_i * dϕ) for z_prime_i in z_prime ]

            @tensor f_dis[k] := -Ω_uu[i,k]*dμ₁[i]
            dϕ += im * get_operator(f_dis,z) * ϕ # -F_dis

            d_state_vec[:,i] .= vcat(dμ₁,real.(dϕ.data),imag.(dϕ.data))

        end
    end

    if seed_input == 0
        return SDEProblem(time_evol_det,time_evol_sto,state_vec0,(0.0,T),noise_rate_prototype=zeros(length(state_vec0),size(vec_EF)[1]))
    else
        return SDEProblem(time_evol_det,time_evol_sto,state_vec0,(0.0,T),noise_rate_prototype=zeros(length(state_vec0),size(vec_EF)[1]),seed=seed_input)
    end

end





function get_hist_μ₁ϕ(sol::Any,Nfocks::Array{Int64,1},Nsave::Int,T::Float64)
    N = length(Nfocks)

    μ₁_hist = Array{Vector{Float64}}(undef,Nsave)
    ϕ_hist = Array{Any}(undef,Nsave)

    for i in 1:Nsave
        μ₁_hist[i] = Vector{Float64}(undef,2N)
        ϕ_hist[i] = ground_state(Nfocks)
        state_vec = sol(T*i/Nsave)
        μ₁_hist[i] = state_vec[1:2N]
        ϕ_hist[i].data[:] .= state_vec[2N+1:2N+length(ϕ_hist[i].data)] + im * state_vec[2N+1+length(ϕ_hist[i].data):2N+length(ϕ_hist[i].data)*2]

    end
    return μ₁_hist,ϕ_hist
end
