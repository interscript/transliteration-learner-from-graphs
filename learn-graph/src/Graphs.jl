

#=========================================================
    LIB FOR NODES
=========================================================#


mutable struct Node

    x::Dict
    children::Union{Vector{Node}, Nothing}

end

import Base.copy

function copy(dict::Dict{String, Any})
    dico = Dict{String, Any}()
    for k in keys(dict)
        dico[k] = isnothing(dict[k]) ? dict[k] : dict[k]
    end
    dico
end

function copy(node::Union{Node, Nothing})
     isnothing(node) ?
        nothing : Node(copy(node.x),
                       isnothing(node.children) ?
                            nothing : [copy(n) for n in node.children])
end


mutable struct Functor

    fct::Function
    meta::Dict{Symbol, Vector{String}}

end


function get_node(id::Int64, df_Nodes::DataFrame, depth::Int64=0)

    filter(row -> row.Id == id, df_Nodes)[1,:] |>
        (x -> (d=Dict(pairs(x));d[:depth]=depth;d))

end


function get_node(interfaceName::String, df_Nodes::DataFrame, depth::Int64=0)

    filter(row -> row.Label == interfaceName, df_Nodes)[1,:] |>
        (x -> (d=Dict(pairs(x));d[:depth]=depth;d))

end


function get_children_ids(id::Int64, df_Arrows::DataFrame)

     df_Arrows[df_Arrows[!,"Line Source"] .== id, :]

end


function createTree(node::Union{Node, Nothing},
                    df_Nodes::DataFrame,
                    df_Arrows::DataFrame,
                    df_Brains::DataFrame,
                    history::Vector{Node} = Node[])

    if node != nothing

        id = node.x[:Id]
        df_ids = get_children_ids(id, df_Arrows)
        node.x[:map] = size(df_ids.Label)[1] > 1 ?
                Dict(map(x -> x[2] => x[1], enumerate(df_ids.Label))) :
                nothing


        if !haskey(dicCODE, node.x[:Label]) &
                !(node.x[:Label] in df_Brains[!, "Label"])

            @warn "unimplemented Node:: Id", node.x[:Id], " Name: " , node.x[:Label]

        end

        if size(df_ids)[1] > 0

            node.children = map(ix ->

                    if ix in [n.x[:Id] for n in history]

                        filter(n -> n.x[:Id] == ix, history)[end]

                    else

                        createTree(
                            Node(get_node(ix, df_Nodes, node.x[:depth]+1),
                                 nothing),
                            df_Nodes, df_Arrows, df_Brains,
                            [history;[node]])

                    end,
                                df_ids[!,"Line Destination"])

        end

    end

    node

end
