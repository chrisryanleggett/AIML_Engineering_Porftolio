#include "core_functions.hpp"
#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
using namespace std;

void printMessage(string text)
{
  cout << "Hello " + text + "!" << endl;
}

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