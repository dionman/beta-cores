from tabulate import tabulate

table = ([["ADULT", 10, 30162, 7413, 3700, (1)],
		  ["TRANSACTIONS", 50, 80000, 4079, 2036, (1)],
		  ["WEBSPAM", 127, 126185, 13789, 6907, (1)]])

headers = ["Dataset", "D", "N", "Ntest", "Pos. test data","Coreset Hyperparams\n ($\gamma_0$)"]

print(tabulate(table, headers, tablefmt="latex"))
