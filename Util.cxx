#include <iostream>

using namespace std;

double clockDiffMs(clock_t end, clock_t start)
{
	double diffTicks = end - start;
	double diffMs = diffTicks / (CLOCKS_PER_SEC/1000);
	return diffMs;
}

void printTimeMs(double time)
{
	cout << time << "ms" << endl;
}

void printTimeMs(string str, double time)
{
	cout << str << time << "ms" << endl;
}
