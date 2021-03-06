library(tidyverse)

titanic = read.csv('./test.csv')

# gets the title of the passenger's name 
getTitle <- function(FullName) {
  title <- gsub('(.*,\\s*)|(\\..*)', '', x=FullName)
  if (title != 'Mr' & title != 'Miss' & title != 'Mrs' & title != 'Master') {
    title <- 'Other'
  }
  title
}

# preparing data
titanic$Title <- sapply(titanic$Name, getTitle)
missing <- titanic %>% filter(is.na(Age))
present <- titanic %>% filter(!is.na(Age))

# create and validate model
model <- lm(Age ~ Pclass + SibSp + Parch + Title, data=present)
missing$Age <- predict(model, newdata=missing)

# create one-hot vector for Embarked, Title
final <- rbind(present, missing) %>% arrange(PassengerId)
final$EmbC <- (final$Embarked == 'C') * 1
final$EmbS <- (final$Embarked == 'S') * 1
final$EmbQ <- (final$Embarked == 'Q') * 1

final$Mr     <- (final$Title == 'Mr') * 1
final$Mrs    <- (final$Title == 'Mrs') * 1
final$Miss   <- (final$Title == 'Miss') * 1
final$Master <- (final$Title == 'Master') * 1
final$Other  <- (final$Title == 'Other') * 1


final$C1 <- (final$Pclass == 1) * 1
final$C2 <- (final$Pclass == 2) * 1
final$C3 <- (final$Pclass == 3) * 1

final <- final %>% filter(!is.na(Embarked))

# Write final dataframe to file
write.csv(final, './test_final.csv', row.names = FALSE)