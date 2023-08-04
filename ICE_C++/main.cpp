#include <vector>
#include <iostream>
#include <chrono>
#include <windows.h>
#include <map>
#include <algorithm>
#include "ice_structure.h"
#include "Unit_test.h"

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str) {
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);

    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, ',')) {
        result.push_back(cell);
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty()) {
        // If there was a trailing comma then add an empty element.
        result.emplace_back("");
    }
    return result;
}

class StatisticsCalculator {
public:
    static void calculateMeanModeMedianRange(const std::vector<double>& data) {
        if (data.empty()) {
            std::cout << "The vector is empty." << std::endl;
            return;
        }

        // Calculate mean
        double sum = 0.0;
        for (double num: data) {
            sum += num;
        }
        double mean = sum / double(data.size());

        // Calculate mode
        std::map<double, int> frequencyMap;
        for (double num: data) {
            frequencyMap[num]++;
        }
        double mode = -1;
        int maxFrequency = 0;
        for (const auto &pair: frequencyMap) {
            if (pair.second > maxFrequency) {
                mode = pair.first;
                maxFrequency = pair.second;
            }
        }

        // Calculate median
        std::vector<double> sortedData = data;
        std::sort(sortedData.begin(), sortedData.end());
        double median;
        int n = int(sortedData.size());
        if (n % 2 == 0) {
            median = (sortedData[n / 2 - 1] + sortedData[n / 2]) / 2.0;
        } else {
            median = sortedData[n / 2];
        }

        // Calculate range
        double minValue = *std::min_element(data.begin(), data.end());
        double maxValue = *std::max_element(data.begin(), data.end());
        double range = maxValue - minValue;

        // Print the results
        std::cout << "Mean: " << mean << std::endl;
        std::cout << "Mode: " << mode << std::endl;
        std::cout << "Median: " << median << std::endl;
        std::cout << "Range: " << range << std::endl;
    }
};

void userprompt(int argc, char** argv) {
    std::vector<Point> data;
    if(argc >= 3) {
        std::ifstream file(argv[1]);

        if (file.is_open()) {
            // skip the columns name
            std::vector<std::string> row;
            while (!(row = getNextLineAndSplitIntoTokens(file)).empty()) {
                if(row.size() != 1) {
                    Point point;
                    for (size_t i = 0; i < row.size() - 1; i++) {
                        point.X.push_back(std::stod(row[i]));
                    }
                    point.Y = int(std::stod(row.back()));
                    data.push_back(point);
                }
                else {
                    break;
                }
            }

            int iterations = 1;
            std::vector<double> time;
            if(argc == 4) {
                iterations = std::abs(int(std::stod(argv[3])));
            }
            else {
                std::cout << "The Optional<iterations> Argument was not Defined" << std::endl;
            }

            Config result;

            for(int k = 0; k < iterations; k++) {

                std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
                SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

                int ub = int(std::stod(argv[2]));
                ICE ice1 = ICE(ub, data);
                ice1.e01gen();

                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

                result = ice1.sel01opt();
                //            ice1.exportModel();

                // Calculate the elapsed time
                std::chrono::duration<double> elapsed_seconds = end - start;
                time.push_back(elapsed_seconds.count());
                ice1.sanityCheck(result);

            }

            // Take the result from the last iteration
            if (!result.model.w.empty()) {
                std::cout << "Model found!" << std::endl;

                // Print the obtained model's weight vector
                std::cout << "Weight vector (w):" << std::endl;
                for (double &f: result.model.w) {
                    std::cout << f << " ";
                }
                std::cout << std::endl;

                // Print the loss of the obtained model
                std::cout << "Loss (l): " << result.model.l << std::endl;

                for (int i: result.comb) {
                    Vector b = data[i].X;
                    for (double a: b) {
                        std::cout << a << ", ";
                    }
                    std::cout << data[i].Y << ", ";
                }
                std::cout << std::endl;

            } else {
                std::cout << "No valid model found." << std::endl;
            }

            std::cout << "Elapsed time: seconds" << std::endl;

            StatisticsCalculator::calculateMeanModeMedianRange(time);

        }
        else {
            std::cout << "The file your provide could not be located Please Try Again" << std::endl;
        }
    }
    else {
        std::cout << "This executable expects 2 parameters " << argc-1 << " were provided." << std::endl;
        std::cout << "<File Location> <Upper Bound> Optional<iterations>" << std::endl;

    }
}

int main(int argc, char** argv) {
    userprompt(argc, argv);
    UnitTest();
    return 0;
}