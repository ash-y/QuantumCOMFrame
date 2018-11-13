using LinearAlgebra
using TensorOperations

####################################################################
#
# construct \Omega matrix
#

function Omega_uu(N::Int)
    id = Matrix{Float64}(I, N, N)
    zs = zeros(N,N)
    return [zs -id; id zs]
end

function Omega_dd(N::Int)
    id = Matrix{Float64}(I, N, N)
    zs = zeros(N,N)
    return [zs id; -id zs]
end


function change_coordinate(X::Array{<:Number},S::AbstractArray{<:Number,2})
    size_X =  size(X)
    N = div(size_X[1], 2)
    rank_X = ndims(X)
    X_prime = zeros(Complex{Float64},size_X...)
    indeces_set = Iterators.product(ntuple(i->1:2N,rank_X)...)

    for inds_after in indeces_set
        for inds_before in indeces_set
            Xprime_element = X[inds_before...];
            for i in 1:rank_X
                Xprime_element = Xprime_element * S[inds_before[i],inds_after[i]]
            end;
            X_prime[inds_after...] += Xprime_element
        end;
    end;
    return X_prime
end


############################################################
#symmetrize tensors
function symmetrize_tensor(X::Array{<:Number})
    rank_X = ndims(X)
    if rank_X ==1; return X; end;

    #construct permutation_group
    permutation_group = Set([])
    inds_for_perms = Iterators.product(ntuple(i->1:(rank_X-i+1),rank_X)...)
    size_perm_set = length(inds_for_perms)
    for perm in inds_for_perms
        perm = collect(perm)
        vec_perm  = zeros(Int64,rank_X) # permutated [1:N]
        #generate an element of the permutation group from perm
        for i in 1:rank_X
            j = 1
            while true
                if perm[i] == 1 && vec_perm[j] == 0; vec_perm[j]=i; break;
                elseif vec_perm[j] != 0; j+=1;
                else j+=1; perm[i]-=1
                end;
            end;
        end;
        push!(permutation_group,vec_perm)
    end;

    indeces_set = Iterators.product(ntuple(i->1:2N,rank_X)...)
    X_sym = zeros(eltype(X),size(X)...)
    for inds in indeces_set
        for perm in permutation_group
            inds_perm = Tuple(permute!([inds...],perm)) #permute inds by perm
            X_sym[inds...] += 1.0 / size_perm_set * X[inds_perm...]
        end;
    end;
    return X_sym
end;

############################################################
# get tensors of the effecitive Hamiltonian including Lindblad terms
#
function get_effective_Hamiltonian_tensors!(A′::Array{Complex{Float64},1},B′::Array{Complex{Float64},2},C′::Array{Complex{Float64},3},D′::Array{Complex{Float64},4},A::Array{Float64,1},B::Array{Float64,2},C::Array{Float64,3},D::Array{Float64,4},vec_EF::Array{Array{Complex{Float64}},2},μ₁::Array{Float64,1},μ₂::Array{Float64,2})

    N = Int(length(A)/2)
    Ω_dd = Omega_dd(N)
    A′ .= A; B′ .= B; C′ .= C; D′ .= D;

    for i in 1:size(vec_EF)[1]
        E = vec_EF[i,1];
        F = vec_EF[i,2];
        real_E = real.(E);
        real_F = real.(F);
        exp_Re_L = dot(real_E,μ₁) + tr(real_F*μ₂)
        @tensor begin
            A′[i]       += im * exp_Re_L * E[i]
            B′[i,j]     += im * exp_Re_L * F[i,j]
            B′[i,j]     -= im *  real_E[i] * E[j]
            C′[i,j,k]   -= im *  real_E[i] * F[j,k]
            C′[i,j,k]   -= im *  real_F[i,j] *E[k]
            D′[i,j,k,l] -= im *  real_F[i,j] * F[k,l]
        end
    end

    return (A′,symmetrize_tensor(B′),symmetrize_tensor(C′),symmetrize_tensor(D′))
end


############################################################
# get raw 2rd-order moments from central moments
#
function get_2nd_raw_moment(μ::Array{<:Number,1},μ_dd::Array{<:Number,2})

    @tensor begin
        μ′_dd[i,j]  := μ_dd[i,j] + μ[i] * μ[j]
    end

    return symmetrize_tensor(μ′_dd)
end
############################################################
# get raw 3rd-order moments from central moments
#
function get_3rd_raw_moment(μ::Array{<:Number,1},μ_dd::Array{<:Number,2},μ_ddd::Array{<:Number,3})

    @tensor begin
        μ′_ddd[i,j,k]  := μ_ddd[i,j,k] + 3.0 * μ_dd[i,j] * μ[k] + μ[i] * μ[j] * μ[k]
    end

    return symmetrize_tensor(μ′_ddd)
end


############################################################
# get raw 4th-order moments from central moments
#
function get_4th_raw_moment(μ::Array{<:Number,1},μ_dd::Array{<:Number,2},μ_ddd::Array{<:Number,3},μ_dddd::Array{<:Number,4})

    @tensor begin
        μ′_dddd[i,j,k,l]  := μ_ddd[i,j,k,l] + 4. * μ_ddd[i,j,k] * μ[l] + 6.0 * μ_dd[i,j] * μ[k] * μ[l] + μ[i] * μ[j] * μ[k]
    end

    return symmetrize_tensor(μ′_dddd)
end


#################################################################################################
#  get symmetrized z coordinate tensors from a coordinate tensors
#
#

function get_sym_tensors_with_z_coord(array_of_tensors::Array{Array{Complex{Float64}},1})

    ret = Array{Array{Float64}}(undef,length(array_of_tensors))
    for i in 1:length(ret)
        ret[i] = real.(symmetrize_tensor(converta2z(array_of_tensors[i])))
    end
    return Tuple(ret)
end
