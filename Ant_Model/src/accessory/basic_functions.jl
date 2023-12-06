using DrWatson
@quickactivate "Ant_Model"
using Interpolations

# periodic version, target index is to the left of index used in rhs
function diffP_left(A::Array{T,N}; dims::Int64, dx::T)::Array{T,N} where T<:Real where N
    sizes = size(A)
    B::Array{T,3} = reshape(A, dims == 1 ? 1 : prod(sizes[(1:dims-1)]), sizes[dims], dims == N ? 1 : prod(sizes[(dims+1:N)]))
    Bsize = size(B)
    s = zeros(T, Bsize)

    for k=1:Bsize[3]
        for j=1:Bsize[1]
            for i=2:Bsize[2]
                s[j,i,k] = (B[j,i,k] - B[j,i-1,k])/dx
            end
            s[j,1,k] = (B[j,1,k] - B[j,Bsize[2],k])/dx
        end
    end

    return reshape(s, sizes)
end

# periodic version, target index is to the right of index used in rhs
function diffP_right(A::Array{T,N}; dims::Int64, dx::Float64)::Array{T,N} where T<:Real where N
    sizes = size(A)
    B::Array{T,3} = reshape(A, dims == 1 ? 1 : prod(sizes[(1:dims-1)]), sizes[dims], dims == N ? 1 : prod(sizes[(dims+1:N)]))
    Bsize = size(B)
    s = zeros(T, Bsize)

    for k=1:Bsize[3]
        for j=1:Bsize[1]
            for i=1:Bsize[2]-1
                s[j,i,k] = (B[j,i+1,k] - B[j,i,k])/dx
            end
            s[j,Bsize[2],k] = (B[j,1,k] - B[j,Bsize[2],k])/dx
        end
    end

    return reshape(s, sizes)
end

function meanP_left(A::Array{T,N}; dims::Int64)::Array{T,N} where T<:Real where N
    sizes = size(A)
    B::Array{T,3} = reshape(A, dims == 1 ? 1 : prod(sizes[(1:dims-1)]), sizes[dims], dims == N ? 1 : prod(sizes[(dims+1:N)]))
    Bsize = size(B)
    s = zeros(T, Bsize)

    for k=1:Bsize[3]
        for j=1:Bsize[1]
            for i=2:Bsize[2]
                s[j,i,k] = (B[j,i,k] + B[j,i-1,k])/2.0
            end
            s[j,1,k] = (B[j,1,k] + B[j,Bsize[2],k])/2.0
        end
    end

    return reshape(s, sizes)
end

# periodic version
function upwindP!(F::Array{T,N}, u::Array{T,N}; dims::Int64=1)::Array{T,N} where T<:Real where N
    sizes = size(F)
    # size(u) = size(F) should be

    B::Array{T,3} = reshape(F, dims == 1 ? 1 : prod(sizes[(1:dims-1)]), sizes[dims], dims == N ? 1 : prod(sizes[(dims+1:N)]))
    a::Array{T,3} = reshape(u, dims == 1 ? 1 : prod(sizes[(1:dims-1)]), sizes[dims], dims == N ? 1 : prod(sizes[(dims+1:N)]))

    Bsize = size(B)
    # Ba = zeros(T, Bsize)

    for i=1:Bsize[1]
        for k=1:Bsize[3]
            for j=2:Bsize[2]
                B[i,j,k] = (B[i,j,k] > 0.0 ? B[i,j,k]*a[i,j-1,k] : B[i,j,k]*a[i,j,k])
            end
            B[i,1,k] = (B[i,1,k] > 0.0 ? B[i,1,k]*a[i,Bsize[2],k] : B[i,1,k]*a[i,1,k])
        end
    end

    return reshape(B, sizes)
end


function polarisation(f::Array{T,3}, Cosθ::Array{T,1}, Sinθ::Array{T,1}; Δθ::Float64)::Array{T,3} where T<:Real
    sizes = size(f);
    p = Array{T,3}(undef, 2, sizes[1], sizes[2])

    for i = 1:sizes[1]
        for j = 1:sizes[2]
            p[1,i,j] = sum(f[i,j,:].*Cosθ) * Δθ
            p[2,i,j] = sum(f[i,j,:].*Sinθ) * Δθ
        end
    end

    return p
end

function centdiff(A::Array{Float64,2}; dims::Int64, dx::Float64)::Array{Float64,2} 
    sizes = size(A)
    s = zeros(Float64, sizes)
    if dims == 1
        for j=1:sizes[2]
            for i=2:sizes[1]-1
                s[i,j] = (A[i+1,j] - A[i-1,j])/(2*dx)
            end
            s[1,j] = (A[2,j] - A[sizes[1],j])/(2*dx)
            s[sizes[1],j] = (A[1,j] - A[sizes[1]-1,j])/(2*dx)
        end
    elseif dims == 2
        for i=1:sizes[1]
            for j=2:sizes[2]-1
                s[i,j] = (A[i,j+1] - A[i,j-1])/(2*dx)
            end
            s[i,1] = (A[i,2] - A[i,sizes[2]])/(2*dx)
            s[i,sizes[2]] = (A[i,1] - A[i,sizes[2]-1])/(2*dx)
        end
    end
    return reshape(s, sizes)
end

function interpolatec(c::Matrix{Float64};x,y,θ,Nx::Int64,Ny::Int64,Nθ::Int64,λ1::Float64,Δx::Float64,Δy::Float64)::Array{Float64,3}
    itp = Interpolations.interpolate(c, BSpline(Linear(Periodic(OnCell())))) # interpolate linearly between the data points
    stp = Interpolations.scale(itp,x,y) # re-scale to the actual domain
    etp = Interpolations.extrapolate(stp, (Periodic(), Periodic()))
    s = Array{Float64,3}(undef,Nx,Ny,Nθ)
    s = [etp(x̃+λ1*cos(θ̃),ỹ+λ1*sin(θ̃)) for x̃ ∈ x,ỹ ∈ y, θ̃ ∈ θ]
    return s
end