rmd_files: 
delete_merged_file: true
  html: ["index.Rmd", "01-intro.Rmd", "ipw_aipw.Rmd", 
  "grf_causal_forest.Rmd", "01heterogenous_treatment.Rmd"]

# bookdown::render_book("index.Rmd", "bookdown::gitbook")
# bookdown::render_book("index.Rmd", "bookdown::pdf_book")
# bookdown::preview_chapter("01-intro.Rmd", "bookdown::gitbook") ## this previews only one chapter, intro chapter in this case

N  <- 2000

fun_ATE  <- function(tau){
    gender  <- rbinom(N, size = 1, p = 0.5)
    race  <- rbinom(N, size = 1, p = 0.5)
    W  <- rbinom(N, size = 1, p = 0.5)    
    Y  <- 50 + tau * W  + gender * 5 + race * 10 + rnorm(n = N, mean = 5, sd = 5)

    tau_hat  <-  mean(Y[which(W == 1)]) - mean(Y[which(W == 0)]) 
    return(tau_hat)
}

print(paste("the ATE estimate is: ", fun_ATE(tau = 10)))

reps  <-  2000
store  <- rep(0, reps)

for(i in 1:reps){
    store[i]  <- fun_ATE(tau = 10)
}


gender  <- rbinom(N, size = 1, p = 0.5)
race  <- rbinom(N, size = 1, p = 0.5)
W  <- rbinom(N, size = 1, p = 0.4 * (gender > 0) + 0.4 * (race > 0)) 
Y  <- 50 + 10 * W  + gender * 5 + race * 10 + rnorm(n = N, mean = 5, sd = 5)   
reg  <- lm(Y ~ W + gender + race)
coefficients(reg)[[2]]


print(paste(c("treated males: ", "treated females: ") , table(gender[W == 1])))
print(paste(c("treated Whites: ", "treated Blacks: ") , table(race[W == 1])))

    
    gender  <- rbinom(N, size = 1, p = 0.5)
    race  <- rbinom(N, size = 1, p = 0.5)
    W  <- rbinom(N, size = 1, p = 0.2 + 0.4 * (gender > 0) + 0.2 * (race > 0))    
    Y  <- 50 + 10 * W  + gender * 5 + race * 10 + rnorm(n = N, mean = 5, sd = 5)


    # female-Blacks
    tau_hat1  <-  mean(Y[which(W == 1 & gender == 1 & race == 1)]) - mean(Y[which(W == 0 & gender == 1 & race == 1)]) 
    w1  <- sum(gender == 1 & race == 1) / N

    # female-Whites
    tau_hat2  <-  mean(Y[which(W == 1 & gender == 1 & race == 0)]) - mean(Y[which(W == 0 & gender == 1 & race == 0)]) 
    w2  <- sum(gender == 1 & race == 0) / N

    # male-Blacks
    tau_hat3  <-  mean(Y[which(W == 1 & gender == 0 & race == 1)]) - mean(Y[which(W == 0 & gender == 0 & race == 1)]) 
    w3  <- sum(gender == 0 & race == 1) / N

    # male-Whites
    tau_hat4  <-  mean(Y[which(W == 1 & gender == 0 & race == 0)]) - mean(Y[which(W == 0 & gender == 0 & race == 0)]) 
    w4  <- sum(gender == 0 & race == 0) / N

    tau_hat  <- tau_hat1 * w1 + tau_hat2 * w2 + tau_hat3 * w3 + tau_hat4 * w4

    return(list(table(gender[W == 1]), table(race[W == 1]), tau_hat))



