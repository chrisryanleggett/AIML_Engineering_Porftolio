#include "core_functions.hpp"
#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
using namespace std;

// Prints a greeting message with the provided text
void printMessage(string text)
{
  cout << "Hello " + text + "!" << endl;
}

// Verifies that the provided username matches the "username" environment variable
bool verifyUser(std::string username)
{
  const char* envUsername = getenv("username");
  if (envUsername == nullptr)
  {
    return false;
  }
  string usernameEnvVar = envUsername;
  return username.compare(usernameEnvVar) == 0;
}