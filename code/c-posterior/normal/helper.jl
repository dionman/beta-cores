# Helper functions

drawnow() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
latex(s) = latexstring(replace(s," ","\\,\\,"))
normpdf(x,m,s) = (r=(x-m)./s; exp.(-0.5*r.*r)./(sqrt(2*pi)*s))

# Color palette from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colors = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080", "#ffe119", "#FFFFFF", "#000000"]
colors[21] = "#07c6b3"  # replace white with turquoise
T10colors = split("ygbrcmk","")
shapes = "osd^v*"

# Compute histogram with the specified bin edges,
# where x[i] is in bin j if edges[j] < x[i] <= edges[j+1].
function histogram(x, edges=[]; n_bins=sqrt(length(x)), weights=ones(length(x)))
    if isempty(edges)
        mn,mx = minimum(x),maximum(x)
        r = mx-mn
        edges = linspace(mn-r/n_bins, mx+r/n_bins, n_bins)
    else
        n_bins = length(edges)-1
    end
    counts = zeros(Float64,n_bins)
    for i=1:length(x)
        for j=1:n_bins
            if (edges[j] < x[i] <= edges[j+1])
                counts[j] += weights[i]
                break
            end
        end
    end
    return counts,edges
end

function plot_histogram(x; titlestring="", edges=[], nbins=50, kwargs...)
    if isempty(edges); xmin,xmax = minimum(x),maximum(x); edges = linspace(xmin-(xmax-xmin)/100, xmax, nbins); end
    counts,edges = histogram(x,edges)
    dx = edges[2]-edges[1]
    bar(edges[1:end-1],(counts/length(x))/dx,dx; kwargs...)
    title(titlestring,fontsize=17)
end


