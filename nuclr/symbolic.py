from pysr import PySRRegressor
# sys.path.append(r'C:\Users\gorth\Dropbox (MIT)\Shared\Papers\AI for nuclear\ai-nuclear-nn_test_long_run\lib')
 
def pysr_fit(X,y):

    model_pysr = PySRRegressor(
        procs=4,
        populations=8,
        # ^ 2 populations per core, so one is always running.
        population_size=60,
        # ^ Slightly larger populations, for greater diversity.
        ncyclesperiteration=10000, 
        # Generations between migrations.
        niterations=100,  # Run forever
        # early_stop_condition=(
        #     "stop_if(loss, complexity) = loss < 0.0001 && complexity < 15"
        #     # Stop early if we find a good and simple equation
        # ),
        timeout_in_seconds=60 * 60 * 24,
        # ^ Alternatively, stop after 24 hours have passed.
        maxsize=30,
        # ^ Allow greater complexity.
        maxdepth=30,
        # ^ But, avoid deep nesting.
        binary_operators=["*","/","+","-"],
        #, "A(x,y) = (abs(x+y))^(2/3)"
        # unary_operators=[
        #     "p2o3(x) = cbrt(square(abs(x)))",
        #     "pm1o3(x) = 1/(cbrt(abs(x)))",
        #     "square",       
        #     # ^ Custom operator (julia syntax)"square","cbrt",
        # ],
        unary_operators=["sin", "cos", "sqrt"],
        #unary_operators=["square", "cube","exp","sin","square","log","sqrt"],
        #unary_operators=["square", "cube","exp","sin","square","tan","tanh","log","sqrt"],
        # constraints={"/": (5,3),
        #               "square": 3,
        #               "sin": 3
        #               },
        # nested_constraints={
        # "sin": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
        # "exp": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
        # "tan": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
        # "tanh": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
        # "log": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0},
        # "sqrt": {"exp":0,"sin":0,"tan":0,"tanh":0,"log":0,"sqrt":0}},
        nested_constraints={"sin": {"sin":3, "cos":3}, "cos": {"cos":3}},
        # constraints={
        # "/": (5,3),
        # # "p2o3": 3,
        # # "pm1o3": 3,
        # # "square": 3
        # },
        # ^ Nesting constraints on operators. For example,
        # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
        # extra_sympy_mappings={"p2o3": lambda x: (abs(x))**(2/3),"pm1o3": lambda x: (abs(x))**(-1/3),
        #                       "inv": lambda x: 1/abs(x)},
        # ^ Define operator for SymPy as well
        complexity_of_constants=1,
        # ^ Punish constants more than variables
        weight_randomize=0.3,
        # ^ Randomize the tree much more frequently
        loss="loss(x, y) = (x - y)^2",
        # ^ Custom loss function (julia syntax)
    )
    X = X.reshape(-1, 1)
    model_pysr.fit(X, y)
    
    return model_pysr
