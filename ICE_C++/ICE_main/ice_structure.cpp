// ICE Class Documentation

/**
 * ICE is a class that represents a program for optimizing models using the ICE (Iterated Conditional Expectations) algorithm.
 * It provides methods for model training, evaluation, and exporting the optimized model.
 */

#include "ice_structure.h"
/**
 * ICE class constructor.
 *
 * This constructor initializes an instance of the ICE class with the provided upper bound
 * 'ub' and dataset 'dxl'. The ICE class is used for certain computations involving the
 * provided data. It initializes various member variables including 'ub', 'D', 'data', 'N',
 * 'x', 'd', and 'configs'. The 'ub' represents the upper bound, and 'dxl' is the dataset
 * used for the computations. The dataset contains a collection of 'Point' instances, and
 * 'Point' is a structure representing data points in the dataset.
 *
 * @param ub The upper bound used for computations.
 * @param dxl The dataset containing data points for computations.
 * @return None.
 */
ICE::ICE (const int& ub, const Dataset& dxl) {
    this->ub = ub;
    this->D = dxl.empty() ? 0 :int(dxl.front().X.size());
    this->data = dxl;
    this->N = dxl.empty() ? 0 :int(dxl.size());
    std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 10);

    Config empty;
    this->configs.emplace_back(empty);
    // this is used to value the loss
    for (const Point& p : this->data) {
        Vector f = p.X;
        f.insert(f.begin(), 1);
        this->x.push_back(f);
    }

    // this is used to calculate the w
    for (const Point& p : this->data) {
        Vector s = p.X;
        s.push_back(1);
        this->d.push_back(s);
    }
};

/**
 * Compute the 0/1 loss between two values.
 *
 * This method calculates the 0/1 loss between two integer values 'z' and 'y'. The 0/1 loss is a
 * metric used to evaluate the accuracy of a binary classification model. It returns 0 if either
 * 'z' or 'y' is 0. Otherwise, it returns 0 if 'z' is equal to 'y', and 1 if 'z' is different
 * from 'y'.
 *
 * @param z The first integer value for comparison.
 * @param y The second integer value for comparison.
 * @return The 0/1 loss between 'z' and 'y' (0 or 1).
 */
int ICE::loss01(const int& z, const int& y) {
    // Implement the loss01 function according to your requirements
    // This is a placeholder implementation
    return (z == 0 || y == 0) ? 0 : (z == y ? 0 : 1);
}

/**
 * Fit the model weights to the given configuration.
 *
 * This method fits the model weights to the provided configuration using Gaussian Elimination.
 * It computes the weights 'w' and inserts them into the model 'config->model' as a new vector.
 * The weights are adjusted by multiplying each element by 1, and a bias term is added by inserting
 * -1 at the beginning of the 'w' vector. The fitted weights are stored in the 'config->model.w'.
 *
 * @param config A pointer to the configuration for which the weights are to be fitted.
 * @return None.
 */
void ICE::fitw(Config* config) {

    Vector w = gaussianElimination(config);
    for(auto &t : w ) t*=1;
    w.insert(w.begin(), -1);
    (*config).model.w = w;
}

/**
 * Perform Gaussian Elimination to solve a system of linear equations.
 *
 * This method takes a matrix 'matrix' as input, performs Gaussian Elimination to solve a system
 * of linear equations, and returns the solution 'w' as a vector. The input 'matrix' is expected
 * to be a square matrix augmented with a column for the right-hand side of the equations. The
 * Gaussian Elimination process involves finding the pivot rows, performing row operations to
 * eliminate variables below the pivot, and finally performing back substitution to obtain the
 * solution 'w' for the system of equations. The solution vector 'w' is returned after all
 * computations.
 *
 * @param matrix The input matrix representing a system of linear equations.
 * @return The solution 'w' vector obtained through Gaussian Elimination.
 */
Vector ICE::gaussianElimination(Config* config) {
    Matrix matrix;
    matrix.reserve((*config).comb.size());
    for (auto &i: (*config).comb) {
        matrix.push_back(d[i]);
    }

    int f = int(matrix.size());
    for (int i = 0; i < f; ++i) {
        // Find pivot row
        int pivotRow = (i);
        for (int j = i + 1; j < f; ++j) {
            if (std::abs(matrix[j][i]) > std::abs(matrix[pivotRow][i])) {
                pivotRow = (j);
            }
        }

        // Swap current row with pivot row
        std::swap(matrix[i], matrix[pivotRow]);

        // Perform row operations to eliminate variables below pivot
        for (int j = (i) + 1; j < f; ++j) {
            double factor = matrix[j][i] / matrix[i][i];
            matrix[j][i] = 0; // Make the element zero
            for (int k = (i) + 1; k < f + 1; ++k) {
                matrix[j][k] -= factor * matrix[i][k];
            }
        }
    }

    // Back substitution to obtain the solution
    Vector w(f);
    for (int i = f - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < f; ++j) {
            sum += matrix[i][j] * w[j];
        }
        w[i] = (matrix[i][f] - sum) / matrix[i][i];
    }

    // Print the solution
    return w;
}


/**
 * Evaluate the weighted sum of two vectors using a model configuration.
 *
 * This method takes two input vectors 'dx' and 'config->model.w' and calculates the weighted
 * sum of their elements. Both vectors must be of the same size. The method evaluates the dot
 * product of 'dx' and 'config->model.w' and stores the result in the variable pointed to by
 * 'sum'. The evaluation involves iterating through the elements of the vectors and performing
 * element-wise multiplication and addition to compute the weighted sum.
 *
 * @param dx The first input vector.
 * @param config A pointer to the model configuration containing the weight vector 'w'.
 * @param sum A pointer to the variable where the computed weighted sum will be stored.
 * @throws std::invalid_argument if 'dx' and 'config->model.w' have different sizes.
 * @return None. The result is stored in 'sum'.
 */
void ICE::evalw(const Vector & dx, Config* config, double * sum) {
    if (dx.size() != config->model.w.size()) {
        throw std::invalid_argument(
                "Error: Incompatible two input vector must be of same size. (e01)");
    }
    for (size_t i = 0; i < config->model.w.size(); i++) {
        (*sum) += dx[i] * config->model.w[i];
    }
}

/**
 * Classify the sum based on a threshold value.
 *
 * This method is used to classify the value pointed to by 'sum' based on a threshold value.
 * The absolute value of 'sum' is compared against 'smalleps'. If the absolute value is less
 * than 'smalleps', 'sum' is set to 0. Otherwise, 'sum' is classified as positive (1) if its
 * value is greater than 0, or negative (-1) if its value is less than 0.
 *
 * @param sum A pointer to the double value to be classified.
 * @return None.
 */
void ICE::pclass(double *sum) const {
    if(std::abs(*sum) < this->smalleps) (*sum) = 0;
    (*sum) = (*sum) == 0?0 : (*sum) > 0 ? 1: -1;
}

/**
 * Evaluate the 0/1 loss for a given model configuration.
 *
 * This method evaluates the 0/1 loss for the provided model configuration 'config'. The 0/1 loss
 * is a metric used to evaluate the performance of a binary classification model. It iterates
 * through 'this->n' data points and computes the weighted sum 'y_' by calling 'evalw()' with the
 * input vector 'this->x[i]' and the model configuration 'config'. Then, it classifies 'y_'
 * using 'pclass()', and calculates the 0/1 loss between the classified value and the true
 * label 'this->data[i].Y'. The computed 0/1 loss is added to the model's loss '(*config).model.l'.
 *
 * @param config A pointer to the model configuration for which the 0/1 loss is to be evaluated.
 * @return None.
 */
void ICE::e01(Config* config) {
    for (int i = 0; i <= this->n; i++) {
        double y_ = 0;
        this->evalw(this->x[i], config, &y_);
        this->pclass(&y_);
        (*config).model.l += this->loss01(int(y_), this->data[i].Y);
    };
}

/**
 * Update the model configuration using a data point.
 *
 * This method updates the model configuration 'config' using a data point. If the model's weight
 * vector 'w' is not empty (exists), it computes the weighted sum 'y_' by calling 'evalw()' with
 * the input vector 'this->x[this->n]' and the model configuration 'config'. Then, it classifies
 * 'y_' using 'pclass()', and calculates the 0/1 loss between the classified value and the true
 * label 'this->data[this->n].Y'. The computed 0/1 loss is added to the model's loss
 * '(*config).model.l'.
 *
 * Note: The sequence will be updated whether or not the model exists, as specified in the Haskell
 * code.
 *
 * @param config A pointer to the model configuration to be updated.
 * @return None.
 */
void ICE::cnfupd1(Config* config) {
    // the sequence will be updated whether or not the model exist given by the haskell code
    if(!(*config).model.w.empty()) {
        double y_ = 0;
        this->evalw(this->x[this->n], config, &y_);
        this->pclass(&y_);
        (*config).model.l += this->loss01(int(y_), this->data[n].Y);
    }
}

/**
 * Update the model configuration and calculate the 0/1 loss.
 *
 * This method updates the model configuration 'config' by adding the current data point index
 * 'this->n' to the 'comb' vector of 'config'. If the model's weight vector 'w' is empty and the
 * size of 'comb' is equal to 'this->D', it determines the 'w' normal vector by calling 'fitw()'
 * with 'config'. Then, it calculates the 0/1 loss using 'e01()' for the updated model
 * configuration 'config'.
 *
 * @param config A pointer to the model configuration to be updated and used for loss calculation.
 * @return None.
 */
void ICE::cnfupd2(Config* config) {
    (*config).comb.push_back(this->n);
    if((*config).model.w.empty() && (int((*config).comb.size()) == this->D)) {
        // determine the w normal vector
        this->fitw(config);
        // calculate the 01 loss for 0 to n for both the positive and negative sense
        this->e01(config);
    }
}

/**
 * Check the feasibility of a model configuration.
 *
 * This method checks the feasibility of the provided model configuration 'config'. It returns
 * 'true' if the size of 'config->comb' is less than or equal to 'this->D', indicating that the
 * model configuration is feasible. Otherwise, it returns 'false' if the size of 'config->comb'
 * exceeds 'this->D', indicating that the model configuration is not feasible.
 *
 * @param config A pointer to the model configuration to be checked for feasibility.
 * @return 'true' if the model configuration is feasible, 'false' otherwise.
 */
bool ICE::feasibe(Config * config) const {
    return int((*config).comb.size()) <= this->D;
}

/**
 * Check the viability of a model configuration.
 *
 * This method checks the viability of the provided model configuration 'config'. It returns 'true'
 * if either the model's weight vector 'w' is empty, or the difference between 'this->n', 'this->D',
 * and 'config->model.l', plus 1, is less than or equal to 'this->ub', or 'config->model.l' is less
 * than or equal to 'this->ub'. If any of these conditions is met, the model configuration is
 * considered viable and 'true' is returned; otherwise, 'false' is returned to indicate that the
 * configuration is not viable.
 *
 * @param config A pointer to the model configuration to be checked for viability.
 * @return 'true' if the model configuration is viable, 'false' otherwise.
 */
bool ICE::viable(Config * config) const {
    return (*config).model.w.empty() || (this->n - this->D) + 1 - (*config).model.l <= this->ub + 1 || (*config).model.l <= this->ub + 1;
}

/**
 * Check if a model configuration should be retained.
 *
 * This method checks if the provided model configuration 'config' should be retained. It calls the
 * 'feasible()' method to check the feasibility of the model configuration and the 'viable()' method
 * to check its viability. If both 'feasible()' and 'viable()' return 'true', indicating that the
 * configuration is both feasible and viable, this method returns 'true', suggesting that the model
 * configuration should be retained. Otherwise, if either 'feasible()' or 'viable()' returns 'false',
 * indicating that the configuration is either not feasible or not viable, this method returns 'false'
 * to suggest that the configuration should not be retained.
 *
 * @param config A pointer to the model configuration to be checked for retention.
 * @return 'true' if the model configuration should be retained, 'false' otherwise.
 */
bool ICE::retain(Config * config) const {
    return this->feasibe(config) && this->viable(config);
}

/**
 * Generate 0/1 losses for the remaining model configurations.
 *
 * This method generates 0/1 losses for the remaining model configurations based on the provided
 * dataset 'data'. It iterates over the 'data' points using the 'n' index, starting from 0. For
 * each iteration, the method empties the temporary location for the next iteration and sets the
 * pointer 'next_available_index' accordingly. Then, it iterates over the remaining viable model
 * configurations stored in 'configs', applying the 'choice()' method for each viable configuration.
 * The 'choice()' method generates choices for the given data point and updates the model
 * configurations accordingly.
 *
 * After processing all viable configurations for the current 'n', the method filters nonviable
 * configurations using the 'filtnonviable()' method to remove configurations that are no longer
 * viable. The process continues until all 'data' points have been processed.
 *
 * @return None.
 */
void ICE::e01gen() {

    int size_data = int(this->data.size());
    for(this->n = 0; this->n < int(size_data); this->n++) {
        // emptying the temporary location for the next iteration
        // iterating over the remaining configuration
        int size = int(this->configs.size());

        for(int i = 0; i < size ; i++) {
            if(this->configs[i].viable) {
                this->choice(int(i));
            }
        }
        std::cout <<"n = " << n << " Configuration = "<< this->configs.size() << std::endl;
    }

}

/**
 * Generates choices for the given data point.
 *
 * This method is responsible for generating choices based on the provided data point,
 * identified by the index 'i'. It uses the configuration 'p' associated with the data point
 * to determine the choices. The generated choices are updated in the result vector. If a
 * choice is determined to be viable, it is added to the result vector as a new configuration.
 * Otherwise, the data point is marked as nonviable and added to the nonviable container.
 *
 * @param i The index of the data point for which choices are to be generated.
 * @return None.
 */
void ICE::choice(int i) {

    Config p = this->configs[i];

    if (!this->configs[i].model.w.empty())  {
        if( viable(&this->configs[i])) {
            this->cnfupd1(&this->configs[i]);
        }
        else {
            this->configs[i].viable = false;
        }
    }

    if (int(p.comb.size()) < this->D) {
        this->cnfupd2(&p);
        if (viable(&p)) {
            this->configs.push_back(p);
        }
    }
}

/**
 * Find the next available model configuration index.
 *
 * This method finds the index of the next available model configuration in the 'nonviable' vector
 * and updates the 'next_available_index' member variable accordingly. If the 'nonviable' vector is
 * empty, it means there are no available configurations, and 'next_available_index' is set to -1.
 *
 * If there are nonviable configurations in the 'nonviable' vector, the method increments the 'pltr'
 * (pointer) member variable and checks if it is within bounds (less than the size of 'nonviable'
 * vector minus 1). If so, it updates 'next_available_index' to the index at 'pltr' in 'nonviable',
 * and marks the corresponding element as unavailable by setting it to -1. This ensures that the same
 * configuration is not used more than once.
 *
 * If 'pltr' is not within bounds, it means there are no more available configurations, and
 * 'next_available_index' is set to -1.
 *
 * Note: The 'nonviable' vector stores indices of model configurations that are no longer viable
 * for further processing.
 *
 * @return None.
 */
void ICE::nextavailable() {
    if(this->nonviable.empty()) {
        this->next_available_index = -1;
    }
    else if(this->pltr < static_cast<long long>(this->nonviable.size()) - 1) {
        this->pltr++;
        this->next_available_index = this->nonviable[pltr];
        this->nonviable[this->pltr] = -1;
    }
    else {
        this->next_available_index = -1;
    }
}

/**
 * Filter out nonviable configurations from the 'nonviable' vector.
 *
 * This method filters out nonviable configurations from the 'nonviable' vector. It uses the
 * erase-remove idiom to remove all occurrences of -1 in the 'nonviable' vector, effectively
 * eliminating nonviable configurations. The method iterates through the 'nonviable' vector
 * and removes any element that matches the value -1, leaving only viable configurations in
 * the vector.
 *
 * Note: In this context, viable configurations are those that have not been removed during
 * the ICE process and are still considered valid for further processing.
 *
 * @return None.
 */
void ICE::filtnonviable() {
    this->nonviable.erase(std::remove_if(this->nonviable.begin(), this->nonviable.end(), [&]( auto viable) {
        return viable == -1;
    }), this->nonviable.end());
}

/**
 * Determine the best model configuration.
 *
 * This method determines the best model configuration between the two provided configurations
 * 'result' and 'c2'. The 'result' configuration is updated based on the following conditions:
 *
 * 1. If the 'result' model's weight vector 'w' is empty, it is updated to 'c2'.
 * 2. If 'c2' model's weight vector 'w' is empty, no action is taken.
 * 3. If the 'result' model's loss 'model.l' is less than 'c2.model.l', no action is taken.
 * 4. Otherwise, 'result' is updated to 'c2'.
 *
 * The 'result' will represent the best model configuration after this method is executed,
 * considering the loss values of both configurations and their existing 'w' values.
 *
 * @param result A pointer to the result model configuration to be updated as the best.
 * @param c2 The second model configuration to be compared with 'result'.
 * @return None.
 */
void ICE::best(Config* result, const Config& c2) {
    if ((*result).model.w.empty()) (*result) = c2;
    else if (c2.model.w.empty()) {}
    else if ((*result).model.l < c2.model.l) {}
    else (*result) = c2;
}

/**
 * Determine the worst model configuration.
 *
 * This method determines the worst model configuration between the two provided configurations
 * 'result' and 'c2'. The 'result' configuration is updated based on the following conditions:
 *
 * 1. If the 'result' model's weight vector 'w' is empty, it is updated to 'c2'.
 * 2. If 'c2' model's weight vector 'w' is empty, no action is taken.
 * 3. If the 'result' model's loss 'model.l' is greater than 'c2.model.l', no action is taken.
 * 4. Otherwise, 'result' is updated to 'c2'.
 *
 * The 'result' will represent the worst model configuration after this method is executed,
 * considering the loss values of both configurations and their existing 'w' values.
 *
 * @param result A pointer to the result model configuration to be updated as the worst.
 * @param c2 The second model configuration to be compared with 'result'.
 * @return None.
 */
void ICE::worst(Config* result, const Config& c2) {
    if ((*result).model.w.empty())
        (*result) = c2;
    else if (c2.model.w.empty()) {}
    else if ((*result).model.l > c2.model.l) {}
    else (*result) = c2;
}

/**
 * Select the optimal model configuration with respect to 0/1 losses.
 *
 * This method selects the optimal model configuration from the 'configs' vector based on the
 * 0/1 losses. It iterates through the 'configs' vector, comparing each configuration with the
 * current best configuration (tracked by 'positive') and the current worst configuration (tracked
 * by 'negative'). The 'best()' method is used to update the 'positive' configuration with the
 * configuration that has the lowest loss, and the 'worst()' method is used to update the 'negative'
 * configuration with the configuration that has the highest loss.
 *
 * After iterating through all configurations, the 'negative.model.l' value is adjusted to represent
 * the loss of the negative sense (complement) to the positive sense. If the negative sense's loss
 * is less than or equal to the positive sense's loss, the method updates the 'negative.model' to
 * have the negative sense's loss and negates the 'w' vector elements. Otherwise, it returns the
 * positive sense's model as the optimal configuration.
 *
 * @return The optimal model configuration with respect to 0/1 losses.
 */
Config ICE::sel01opt() {
    Config positive = this->configs.front();
    Config negative = this->configs.front();

    for (const auto & config : this->configs) {
        best(&positive, config);
        worst(&negative, config);
    }

    negative.model.l = (this->N - this->D) - negative.model.l;

    if(negative.model.l <= positive.model.l) {
        Model m;
        m.l = negative.model.l;
        for(auto &t: negative.model.w) m.w.push_back(t * -1);
        negative.model = m;

        if(negative.model.l <= this->ub)
            return negative;
    }

    if(positive.model.l <= this->ub)
        return positive;
    else
        return {};
}


/**
 * Perform a sanity check and output the results to a CSV file.
 *
 * This method performs a sanity check on the provided solution model configuration 'solution'.
 * It checks if the solution has a model attached to it. If the model's weight vector 'w' is empty,
 * the method throws an 'std::invalid_argument' exception indicating that the configuration is
 * invalid and does not have a model attached.
 *
 * If the sanity check passes, the method creates a CSV file named "sanityCheck.csv" and writes the
 * results of the sanity check to the file. The file will contain dynamic columns corresponding to
 * the input features (x0, x1, x2, ..., xn), followed by "y" (true label), "osY_" (predicted label),
 * and "loss" (0/1 loss) for each data point in the 'data' vector.
 *
 * The method calls 'evalw()' to predict the labels using the model configuration 'solution' and
 * 'pclass()' to classify the prediction result. It then calculates the 0/1 loss using 'loss01()'
 * and writes the results to the CSV file for each data point in the 'data' vector.
 *
 * @throws std::invalid_argument if the provided 'solution' configuration has no model attached.
 * @return None.
 */
void ICE::sanityCheck(Config result) {

    if(result.model.w.empty()) {
        throw std::invalid_argument("Error: the provided configuration has no model attached to it");
    }
    std::ofstream file;
    file.open ("sanityCheck.csv");

    // construct the dynamic columns
    for (int i = 0; i < int(this->data.front().X.size()); i++) {
        file << "x" << i <<  ", ";
    }

    file << "y,  assignment, loss\n";

    for (int i = 0; i < int(this->data.size()); i++)  {
        for (double h: this->data[i].X) {
            file << h << ", ";
        }
        file << this->data[i].Y << ", ";
        double y_ = 0;
        this->evalw(this->x[i], &result, &y_);
        this->pclass(&y_);

        if(y_ == 0) {
            y_ = this->data[i].Y;
        }
        file << y_ << ", ";
        file << this->loss01((int(y_)), this->data[i].Y) << "\n";

    }
    file.close();
}

/**
 * Export the ICE model configurations to a CSV file.
 *
 * This method exports the ICE model configurations to a CSV file named "models.csv". The file will
 * contain columns for the model losses and weight vectors, as well as the input features and true
 * labels for each data point in the 'data' vector.
 *
 * The method constructs the dynamic columns for the weight vectors and input features and writes
 * the column names to the CSV file. It then iterates through each 'config' in the 'configs' vector,
 * and for each viable configuration (non-empty weight vector), it writes the loss value and weight
 * vector to the file. It also writes the input features and true label for each data point indexed
 * by 'config.comb' in the 'data' vector.
 *
 * Note: The 'config.comb' vector contains the indices of data points relevant to the corresponding
 * model configuration.
 *
 * @return None.
 */
void ICE::exportModel() {
    std::ofstream file;
    std::string s = "models.csv";
    file.open (s);
    file << "loss ,";

    // construct the dynamic columns
    for (int i = 0; i < this->D + 1; i++) {
        file << "w" << i <<  ", ";
    }

    for(int k = 0; k < this->D; k++) {
        for (int i = 0; i < int(this->x.front().size()); i++)
            file << "x" << i << ", ";
        file << "y" << ", ";
    }
    file << "\n";

    for(const Config& config: this->configs) {
        if (!config.model.w.empty()) {
            file << config.model.l << ", ";
            for (double vector1: config.model.w) {
                file << vector1 << ", ";
            }
            for (auto i : config.comb) {
                std::vector<double> q = this->x[i];
                for (double a: q) {
                    file << a << ", ";
                }
                file << this->data[i].Y << ", ";
            }
            file << "\n";
        }

    }
    file.close();
}