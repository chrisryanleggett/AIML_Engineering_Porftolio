/*
 * This C++ program demonstrates username authentication and serves as a foundation
 * for concurrent programming module. It works by reading a username from a .user
 * file, storing it as an environment variable, accepting an optional command-line
 * username argument, and verifying the username matches both the file value and
 * a hardcoded constant. If authentication succeeds, it prints a greeting message.
 */
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include "core_functions.hpp"
using namespace std;

const string USERNAME = "testuser";

int main(int argc, char *argv[])
{
  int results = 1;
  string username = "";

  // Read username from .user file
  ifstream fin(".user");
  if (!fin)
  {
    cout << "Error opening file. Shutting down..." << endl;
    return 1;
  }

  char *usernameFromFile;
  for (std::string line; getline(fin, line, '\n');)
  {
    usernameFromFile = &line[0];
  }

  fin.close();
  setenv("username", usernameFromFile, 1);

  // Get username from command-line arguments if provided
  if (argc > 1)
  {
    username = argv[1];
  }

  // Verify username against file value, then against hardcoded constant
  bool validUser = verifyUser(username);
  setenv("username", &USERNAME[0], 1);
  validUser = validUser && verifyUser(username);

  // Output result based on authentication
  if (validUser)
  {
    printMessage(username);
    results = 0;
  }
  else
  {
    cout << "Error your usernames don't match check code and .user file." << endl;
  }
  return results;
}