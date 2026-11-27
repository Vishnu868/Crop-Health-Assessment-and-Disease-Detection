import 'dart:io';
import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final String path;
  final Map<String, dynamic> data;
  const ResultScreen({super.key, required this.path, required this.data});

  Widget _tile(IconData icon, String title, String body) =>
      Card(child: ListTile(leading: Icon(icon), title: Text(title), subtitle: Text(body)));

  @override
  Widget build(BuildContext context) {
    final conf = (100 * (data['confidence'] as num)).toStringAsFixed(1);
    final health = (data['health']['health_score'] as num).toDouble();
    return Scaffold(
      appBar: AppBar(title: const Text('Diagnosis')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        ClipRRect(borderRadius: BorderRadius.circular(12), child: Image.file(File(path), height: 220, fit: BoxFit.cover)),
        const SizedBox(height: 16),
        Text(data['disease'], style: Theme.of(context).textTheme.headlineSmall),
        Text('Confidence $conf %'),
        if (data['low_confidence'] == true)
          const Text('Low confidence: retake the photo', style: TextStyle(color: Colors.orange)),
        const SizedBox(height: 12),
        Text('Leaf health ${health.toStringAsFixed(0)}/100'),
        LinearProgressIndicator(value: health / 100),
        const SizedBox(height: 12),
        _tile(Icons.coronavirus_outlined, 'Symptoms', data['symptoms']),
        _tile(Icons.grass, 'Fertilizer', data['fertilizer']),
        _tile(Icons.science_outlined, 'Pesticide', data['pesticide']),
        _tile(Icons.health_and_safety_outlined, 'Precautions', data['precautions']),
      ]),
    );
  }
}
