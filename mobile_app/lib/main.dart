// AgroDoc Flutter client (Material 3). Source only: not built in the rebuild environment.
import 'package:flutter/material.dart';
import 'screens/home.dart';

// Change to your Flask server address (use 10.0.2.2 for the Android emulator).
const String apiBase = 'http://10.0.2.2:5000';

void main() => runApp(const AgroDocApp());

class AgroDocApp extends StatelessWidget {
  const AgroDocApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'AgroDoc',
        theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.green),
        darkTheme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.green, brightness: Brightness.dark),
        home: const HomeScreen(),
      );
}
