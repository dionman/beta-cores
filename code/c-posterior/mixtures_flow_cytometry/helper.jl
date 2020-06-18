# Helper functions

drawnow() = (pause(0.001); get_current_fig_manager()[:window][:raise_]())
latex(s) = latexstring(replace(s," ","\\,\\,"))

# Color palette from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colors = ["#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080", "#ffe119", "#FFFFFF", "#000000"]
colors[21] = "#07c6b3"  # replace white with turquoise
T10colors = split("krbgym","")
T10colors[4] = "#28c133" #"#35e041"
shapes = "osd^v*"

# Draw scatterplots of all pairs of columns in x, with dots colored by the labels assigned to the rows of x,
# and title the plots according to the varnames of the column variables.
function pairwise_scatterplots(x,labels,varnames,colors;kwargs...)
    subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.25)
    L = unique(labels)
    m = size(x,2)
    for i in 1:m, j in 1:m
        subplot(m,m,sub2ind((m,m),j,i))
        for l in L
            plot(x[labels.==l,j],x[labels.==l,i],".",color=colors[l];kwargs...)
        end
        title("$(varnames[i]) vs $(varnames[j])")
    end
end

# Compute the "one-hot" representation corresponding to given the sequence of labels.
function one_hot(labels)
    u = unique(labels)
    K = length(u)
    D = Dict(zip(u,1:K))
    A = Int[(D[l]==k) for l in labels, k=1:K]
    return A,u
end

# Compute the F-measure between two sets of cluster assignments.
# Following Fig 1 of:
#     Rosenberg and Hirschberg, "V-Measure: A conditional entropy-based external cluster evaluation measure."
#     https://academiccommons.columbia.edu/catalog/ac:163494
# Inputs:
#   a = true labels
#   b = inferred labels
function F_measure(a,b)
    A,ua = one_hot(a)  # one-hot representation
    B,ub = one_hot(b)
    nA,nB,n = vec(sum(A,1)),vec(sum(B,1)),sum(A)
    C = A'*B  # intersection counts: C[i,j] = number of items with class i in a and class j in b
    R = C./nA
    P = C./nB'
    F = (2*R.*P)./(R + P + eps())
    f = sum((nA/n).*maximum(F,2))
    return f,F,ua,ub
end

# Find permutation of labels of b to try to match a
function match_labels(a,b)
    f,F,ua,ub = F_measure(a,b)
    m,n = size(F)
    I = zeros(Int,n)
    for j in sortperm(maximum(F,1)[:],rev=true)
        S = setdiff(sortperm(F[:,j],rev=true),I)
        if isempty(S); break; end
        I[j] = S[1]
    end
    I[I.==0] = setdiff(1:n,I)[1:sum(I.==0)]
    Ua = [ua; setdiff(1:n,ua)]
    D = Dict(zip(ub,Ua[I]))
    b_remap = [D[l] for l in b]
    return b_remap
end
