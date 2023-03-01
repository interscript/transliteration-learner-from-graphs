

OPENAIKEY = "<OPENAIKEY>"
DIR = "src/snippets/"


using PyCall


openai = pyimport("openai")
openai.api_key = OPENAIKEY


"query chat gpt"
function queryChatgpt(prompt, 
                      engine="text-davinci-003")
    mssgs = openai.Completion[:create](
        engine = engine,
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        #stop = None,
        temperature = 0.00
    )
    mssgs["choices"][1]["text"]
end


"preprocess command for chat gpt"
function preprocessCommandGPT(command, 
                              fileName="botai/initTMP.txt")

    functionName = join(split(command), "_")
    println(fileName)          
    txt = read(fileName, String)
    txtTMP = replace(txt, "\$command" => command)
    txtTMP = replace(txtTMP, "\$name" => functionName)
    txtTMP

end


"generate function with chat gpt"
function generateFunctionWithChatGPT(command,
                                     fileName="botai/initTMP.txt", 
                                     dir=DIR, 
                                     engine="text-davinci-003")
    

    functionName = join(split(command), "_")

    txtTMP = preprocessCommandGPT(command, fileName)
    # println(txtTMP)
    fileNamePython = dir * functionName * ".py"
    fileNameJulia = dir * functionName * ".jl"

    # println("##############################################")

    if !isfile(fileNamePython)
    
        answerStr = queryChatgpt(txtTMP, engine)
        # println(answerStr)
        open(fileNamePython, "w") do file
            write(file, answerStr)
        end
        println("created code snippets for command: " * command)
    
    end

    if !isfile(fileNameJulia)

        # read python file
        pythonCode = read(fileNamePython, String)
        # convert to julia by asking gpt-3 to do it
        query = "convert the following python code to julia: \n" * pythonCode
        juliaCode = queryChatgpt(query, engine)
        # write to file
        open(fileNameJulia, "w") do file
            write(file, juliaCode)
        end
        println("translated code snippets for command: " * command * " to julia") 

    end

end
