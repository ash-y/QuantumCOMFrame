using PyPlot
using PyCall
import Base64
@pyimport matplotlib.animation as anim



function plot_wigner(ψ)
    xlim = sqrt(size(ψ.data)[1])
    xvec = LinRange(-xlim, xlim, 100) |> collect
    wig = wigner(ψ, xvec, xvec)
    maxwig = maximum(wig)
    imshow(permutedims(wig), vmax=maxwig, vmin=-maxwig, cmap="RdBu",
           extent=[-xlim, xlim, -xlim, xlim], origin="lower")
    colorbar()
end


function plot_wigner(ψ,xlim::Number)
    xvec = LinRange(-xlim, xlim, 100) |> collect
    wig = wigner(ψ, xvec, xvec)
    maxwig = maximum(wig)
    imshow(permutedims(wig), vmax=maxwig, vmin=-maxwig, cmap="RdBu",
           extent=[-xlim, xlim, -xlim, xlim], origin="lower")
    colorbar()
end


function plot_wigner_without_cb(ψ,xlim::Number)
    xvec = LinRange(-xlim, xlim, 100) |> collect
    wig = wigner(ψ, xvec, xvec)
    maxwig = maximum(wig)
    imshow(permutedims(wig), vmax=maxwig, vmin=-maxwig, cmap="RdBu",
           extent=[-xlim, xlim, -xlim, xlim], origin="lower")
end


function showanim(filename)
    base64_video = Base64.base64encode(open(filename))
    display("text/html", """<video controls src="data:video/x-m4v;base64,$base64_video">""")
end



function draw_anim_two_wigners(filename,val_frames::Int,val_interval::Int,ψs,ϕs,xlim::Number)

    function make_frame(i)
        frame_number = i * (Int(floor(length(ψs)/val_frames))-1)+1
        clf()
        subplot(121)
        plot_wigner(ψs[frame_number],xlim)
        subplot(122)
        plot_wigner(ϕs[frame_number],xlim)
    end

    fig = figure(figsize=(12,6))
    withfig(fig) do
        myanim = anim.FuncAnimation(fig, make_frame, frames=val_frames, interval=val_interval)
        myanim[:save](filename, bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    end
    showanim(filename)
end



function draw_anim_wigner(filename,val_frames::Int,val_interval::Int,ψs,xlim::Number)

    function make_frame(i)
        frame_number = i+1
        clf()
        plot_wigner(ψs[frame_number],xlim)
    end

    fig = figure(figsize=(6,6))
    withfig(fig) do
        myanim = anim.FuncAnimation(fig, make_frame, frames=val_frames, interval=val_interval)
        myanim[:save](filename, bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    end
    showanim(filename)
end

function draw_anim_wigner(filename,val_frames::Int,val_interval::Int,ψs,xlim::Number,cb::Bool)

    function make_frame(i)
        frame_number = i+1
        clf()
        if cb == true
            plot_wigner(ψs[frame_number],xlim)
        else
            plot_wigner_without_cb(ψs[frame_number],xlim)
        end

    end

    fig = figure(figsize=(6,6))
    withfig(fig) do
        myanim = anim.FuncAnimation(fig, make_frame, frames=val_frames, interval=val_interval)
        myanim[:save](filename, bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    end
    showanim(filename)
end
