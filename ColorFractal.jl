import REPL
using REPL.TerminalMenus



const coloring_methods = ["manual nodes", "auto peaks"]

function piecewise_linear_range(nodes :: AbstractArray{Float64}) :: Function 
    
    n = length(nodes)
    
    function fun(x)

        # linear search the nodes
        i = 0
        while i < n && x > nodes[i+1]
            i += 1
        end 

        if i == 0
            return 0.0
        end

        if i == n
            return 1.0
        end

        # interpolate 
        Float64(i-1 + (x - nodes[i]) / (nodes[i+1] - nodes[i])) / (n-1)
    end
end

# wizard for creating coloring functions
function wizard() :: Function
    method_menu = RadioMenu(coloring_methods)
    selected_method = coloring_methods[request("Select coloring method:", method_menu)]

    if selected_method == "manual nodes"

        # get number of nodes
        println("Enter desired number of nodes:")
        n_nodes = parse(Int64, readline())
        @assert n_nodes > 0 "Number of nodes must be non-negative."

        # get nodes from user
        nodes = zeros(n_nodes)
        i = 1
        while true
            println("Enter node $i:")
            nodes[i] = parse(Float64, readline())
            if i > 1 && nodes[i] <= nodes[i-1]
                println("Nodes must monotonically increase!")
            else
                i += 1
            end
            if i > n_nodes
                break
            end
        end

        # convert nodes to piecewise function
        piecewise_linear_range(nodes)
        
    elseif selected_method == "auto peaks"
        error("Not implemented!")
    else
        error("No coloring method selected.")
    end
end