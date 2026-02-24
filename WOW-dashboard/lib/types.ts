/**
 * Shared TypeScript types for the dashboard.
 */

export interface ScheduleMetrics {
  overall_risk_score: number;
  expected_overdue_rate: number;
  workload_imbalance_score: number;
  congestion_score: number;
  total_assignments: number;
  avg_predicted_hours: number;
  median_predicted_hours: number;
}

export interface AssignmentPrediction {
  assignment_id: string;
  pred_completion_hours: number;
  risk_score: number;
  top_factors: string[];
}

export interface ChartData {
  risk_histogram: Record<string, unknown>[];
  risk_by_district: Record<string, unknown>[];
  workload_by_engineer: Record<string, unknown>[];
  risk_by_department: Record<string, unknown>[];
}

export interface PredictResponse {
  schedule_metrics: ScheduleMetrics;
  assignment_predictions: AssignmentPrediction[];
  charts: ChartData;
  metadata: Record<string, unknown>;
}
