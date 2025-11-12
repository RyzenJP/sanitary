<?php
/**
 * Automatic AI Training API
 * Triggers AI model retraining via web interface
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');
header('Access-Control-Allow-Headers: Content-Type');

include '../../db.php';
session_start();

// Check if user is authorized (admin or field worker)
if (!isset($_SESSION['user_id']) || !in_array($_SESSION['role'], ['admin', 'field_worker'])) {
    http_response_code(403);
    echo json_encode([
        'success' => false,
        'error' => 'Unauthorized access'
    ]);
    exit;
}

function checkNewData($conn) {
    try {
        // Check for data in last 24 hours
        $query = "
            SELECT COUNT(*) as new_records FROM (
                SELECT reading_time FROM tds_readings 
                WHERE reading_time >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                UNION ALL
                SELECT reading_time FROM turbidity_readings 
                WHERE reading_time >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            ) as combined_data
        ";
        
        $result = $conn->query($query);
        $row = $result->fetch_assoc();
        
        return [
            'new_records' => intval($row['new_records']),
            'needs_training' => intval($row['new_records']) >= 10
        ];
        
    } catch (Exception $e) {
        return [
            'new_records' => 0,
            'needs_training' => false,
            'error' => $e->getMessage()
        ];
    }
}

function triggerTraining() {
    try {
        $ai_dir = __DIR__ . '/../ai/';
        $python_path = 'python'; // Adjust if needed
        
        // Run training script
        $command = "cd \"$ai_dir\" && $python_path auto_train_scheduler.py --force 2>&1";
        $output = shell_exec($command);
        
        return [
            'success' => true,
            'message' => 'Training initiated successfully',
            'output' => $output
        ];
        
    } catch (Exception $e) {
        return [
            'success' => false,
            'error' => $e->getMessage()
        ];
    }
}

function getTrainingStatus() {
    try {
        $log_file = __DIR__ . '/../ai/auto_training.log';
        
        if (!file_exists($log_file)) {
            return [
                'status' => 'No training history',
                'last_training' => null
            ];
        }
        
        // Get last 10 lines of log
        $lines = file($log_file);
        $recent_lines = array_slice($lines, -10);
        
        return [
            'status' => 'Log available',
            'recent_logs' => array_map('trim', $recent_lines),
            'log_file_size' => filesize($log_file)
        ];
        
    } catch (Exception $e) {
        return [
            'status' => 'Error reading logs',
            'error' => $e->getMessage()
        ];
    }
}

// Main API logic
try {
    $action = $_GET['action'] ?? $_POST['action'] ?? '';
    
    switch ($action) {
        case 'check_data':
            $result = checkNewData($conn);
            echo json_encode([
                'success' => true,
                'data' => $result
            ]);
            break;
            
        case 'trigger_training':
            $result = triggerTraining();
            echo json_encode($result);
            break;
            
        case 'status':
            $result = getTrainingStatus();
            echo json_encode([
                'success' => true,
                'data' => $result
            ]);
            break;
            
        case 'info':
            echo json_encode([
                'success' => true,
                'info' => [
                    'automatic_training' => 'Available',
                    'check_interval' => '6 hours',
                    'min_records_for_training' => 10,
                    'supported_actions' => [
                        'check_data' => 'Check for new sensor data',
                        'trigger_training' => 'Force start training',
                        'status' => 'Get training status and logs'
                    ]
                ]
            ]);
            break;
            
        default:
            echo json_encode([
                'success' => false,
                'error' => 'Invalid action. Use: check_data, trigger_training, status, or info'
            ]);
    }
    
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Server error: ' . $e->getMessage()
    ]);
}
?>
