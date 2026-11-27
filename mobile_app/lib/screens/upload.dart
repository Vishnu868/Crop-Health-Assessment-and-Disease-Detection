import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../main.dart';
import 'result.dart';

class UploadScreen extends StatefulWidget {
  final String path;
  const UploadScreen({super.key, required this.path});
  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  String? error;

  @override
  void initState() {
    super.initState();
    _send();
  }

  Future<void> _send() async {
    try {
      final req = http.MultipartRequest('POST', Uri.parse('$apiBase/predict'))
        ..files.add(await http.MultipartFile.fromPath('image', widget.path));
      final res = await http.Response.fromStream(await req.send()).timeout(const Duration(seconds: 30));
      final data = jsonDecode(res.body) as Map<String, dynamic>;
      if (res.statusCode != 200) throw Exception(data['error']);
      if (!mounted) return;
      Navigator.pushReplacement(context,
          MaterialPageRoute(builder: (_) => ResultScreen(path: widget.path, data: data)));
    } catch (e) {
      setState(() => error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Analysing…')),
        body: Center(
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Image.file(File(widget.path), height: 220),
            const SizedBox(height: 24),
            error == null ? const CircularProgressIndicator() : Text('Error: $error'),
          ]),
        ),
      );
}
