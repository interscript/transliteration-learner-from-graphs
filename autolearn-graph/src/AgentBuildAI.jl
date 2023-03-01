

function buildNodeAI(node::Node,
                     dicBRAINS::Dict{String, Node},
                     dfNodes::DataFrame,
                     data::Union{Nothing, Any},
                     dicCodeSnippets::Dict{String, Functor})

    command = node.x[:Label]
    @info "data::> ", data

    name = join(split(command), "_")
    functionName = "lambda_" * name * "_function"
    fileNameJulia = DIR * name * ".jl"
    # println("@@@@@@@@@@@@@@@@@@@@@@@ ", fileNameJulia)
    # if a file with name fileNameJulia exists
    if !isfile(fileNameJulia)

        # generate function with chat gpt
        generateFunctionWithChatGPT(command,
                                    "botai/initSimpleTMP.txt")
    end
        
    # load newly created code snippet
    include(fileNameJulia)
    # update dicCODE
    dicCODE[command] =
        Functor((d,e=nothing,f=nothing) ->
            (f = Symbol(functionName);
             d["txt"] = @eval $f(d); 
             d),
            Dict(:in => ["txt"], :out => ["txt"]))
    
    if haskey(dicCODE, command)

        states = collect(keys(data))
        unrecFields =
               filter(s -> !(s in states), dicCODE[command].meta[:in])

        if unrecFields != String[]

            @error string("command :: ", command,
                          " fields not recognised:: ", unrecFields)
            return nothing
        end

        dicCODE[command].fct(data, dicBRAINS, dfNodes)

    else

        @error string("command not found/built :: ", command)
        nothing

   end

end


function buildAgentAI(node::Node,
                  dicBRAINS::Dict{String, Node},
                  dfNodes::DataFrame,
                  data::Union{Nothing, Any},
                  dicCodeSnippets::Dict{String, Functor})

    name = node.x[:Label]

    println(node.x)

    node =

        if haskey(dicBRAINS, name)

            @info "brain name ::> ", name

            if data["brain"] != name

                # run elsewhere in graph
                buildAgentAI(dicBRAINS[name].children[1],
                         dicBRAINS, dfNodes, data, dicCodeSnippets)
                data["brain"] = name

            end

            if isnothing(node.children)

                @goto __OUT

            end

            # continue locally
            node.children[1]

        else

            @info "node::> ", name
            data = buildNodeAI(node, dicBRAINS, dfNodes, data, dicCodeSnippets)

            if node.children == nothing

                @label __OUT
                return haskey(data, "res") ?
                    data["res"] : data #["word"]

            end

            if length(node.children) > 1

                state = data["state"]
                @info "response::> ", state
                id = node.x[:map][state]
                node.children[id]

            else

                node.children[1]

            end

        end

    @info "data::> ", data
    buildAgentAI(node, dicBRAINS, dfNodes, data, dicCodeSnippets)

end
