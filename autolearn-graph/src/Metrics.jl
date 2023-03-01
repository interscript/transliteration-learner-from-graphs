
function evaluation(trans_orig, trans_model)

    l_bugs = []
    tp, fp = 0, 0

    for (i, (o, t)) in enumerate(zip(trans_orig, trans_model))

        l_orig = filter(s -> s != "", split(strip(o), (' ','_',',','.','!',':',';','?')))
        l_trans = filter(s -> s != "", split(strip(t), (' ','_',',','.','!',':',';','?')))
            
        correct = true
        for d in zip(l_orig, l_trans)
            
            if d[1] == d[2]
                tp += 1
            else
                fp += 1
                correct = false
            end

        end

        if !correct
            push!(l_bugs, Dict("id" => i, "trans" => o, "trans_model" => t))
        end

    end

    println("words accuracy: ", tp / (tp + fp))
    [d["id"] for d in l_bugs]

end
