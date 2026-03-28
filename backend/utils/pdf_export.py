from xhtml2pdf import pisa
import os

def export_to_pdf(seating_plan, filename='seating_plan.pdf', output_folder='.'):
    """
    Export seating plan to PDF
    
    Args:
        seating_plan: The seating plan data
        filename: Name of the PDF file (default: 'seating_plan.pdf')
        output_folder: Folder to save the PDF (default: current directory)
    
    Returns:
        str: Path to the generated PDF file
    """
    # Create better formatted HTML with larger fonts
    html = '''
    
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2d3a4b; text-align: center; margin-bottom: 30px; }
            h2 { color: #515ada; margin-top: 30px; margin-bottom: 15px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
            td { 
                border: 2px solid #ddd; 
                padding: 6px; 
                text-align: center; 
                vertical-align: middle;
                min-width: 130px;
                min-height: 70px;
            }
            .student { background-color: #f8f9fa; font-size: 12px; line-height: 1.1; }
            .empty { background-color: #e9ecef; color: #6c757d; font-size: 12px; }
            .roll { font-weight: bold; color: #2d3a4b; font-size: 14px; }
            .name { color: #495057; font-size: 12px; }
            .branch { color: #6c757d; font-size: 12px; }
            .year { color: #868e96; font-size: 11px; }
        </style>
    </head>
    <body>
        <h1>Exam Seating Plan</h1>
    
    '''
    
    for room_index, room in enumerate(seating_plan):
        room_name = (
            room.get('room_name')
            or room.get('room_number')
            or room.get('name')
            or room.get('room')
            or f'Room {room_index + 1}'
        )

        capacity_raw = room.get('capacity', 'N/A')
        try:
            capacity_value = float(capacity_raw)
            capacity_display = int(capacity_value) if capacity_value.is_integer() else capacity_value
        except (TypeError, ValueError):
            capacity_value = None
            capacity_display = 'N/A'

        students_count = room.get('students_count')
        if students_count is None:
            students_count = 0
            for row in room.get('seats', []):
                for desk in row:
                    if not desk:
                        continue
                    if isinstance(desk, list):
                        students_count += len(desk)
                    elif isinstance(desk, dict) and 'students' in desk:
                        students_count += len(desk.get('students', []))
                    else:
                        students_count += 1

        if capacity_value and capacity_value > 0:
            utilization = f"{(students_count / capacity_value) * 100:.1f}%"
        else:
            utilization = 'N/A'

        building = room.get('building', 'Main Building')

        html += f'''
        <div class="room-card">
            <h2>Room: {room_name}</h2>
            <div class="room-info">
                Building: {building} | Capacity: {capacity_display} | Students: {students_count} | Utilization: {utilization}
            </div>
            <table>
        '''
        
        # Process seats
        for row in room['seats']:
            html += '<tr>'
            for desk in row:
                if desk is not None:
                    # Check if desk contains multiple students (list) or single student (dict)
                    if isinstance(desk, list):
                        # Multiple students per desk
                        html += '<td class="student">'
                        for i, student in enumerate(desk):
                            if i > 0:
                                html += '<hr class="student-separator">'
                            
                            roll_number = student.get('roll_number', 'N/A')
                            name = student.get('name', student.get('Name', 'Unknown'))
                            branch = student.get('branch', student.get('Branch', 'N/A'))
                            year = student.get('year', 'N/A')
                            
                            html += f'''
                            <div style="margin-bottom: 3px;">
                                <div class="roll">{roll_number}</div>
                                <div class="name">{name}</div>
                                <div class="branch">{branch}</div>
                                <div class="year">Year: {year}</div>
                            </div>
                            '''
                        html += '</td>'
                    elif isinstance(desk, dict) and 'students' in desk:
                        # Desk with students array
                        html += '<td class="student">'
                        for i, student in enumerate(desk['students']):
                            if i > 0:
                                html += '<hr class="student-separator">'
                            
                            roll_number = student.get('roll_number', 'N/A')
                            name = student.get('name', student.get('Name', 'Unknown'))
                            branch = student.get('branch', student.get('Branch', 'N/A'))
                            year = student.get('year', 'N/A')
                            
                            html += f'''
                            <div style="margin-bottom: 3px;">
                                <div class="roll">{roll_number}</div>
                                <div class="name">{name}</div>
                                <div class="branch">{branch}</div>
                                <div class="year">Year: {year}</div>
                            </div>
                            '''
                        html += '</td>'
                    else:
                        # Single student per desk
                        roll_number = desk.get('roll_number', 'N/A')
                        name = desk.get('name', desk.get('Name', 'Unknown'))
                        branch = desk.get('branch', desk.get('Branch', 'N/A'))
                        year = desk.get('year', 'N/A')
                        
                        html += f'''
                        <td class="student">
                            <div class="roll">{roll_number}</div>
                            <div class="name">{name}</div>
                            <div class="branch">{branch}</div>
                            <div class="year">Year: {year}</div>
                        </td>
                        '''
                else:
                    html += '<td class="empty">Empty</td>'
            html += '</tr>'
        
        html += '''
            </table>
        </div>
        '''
    
    html += '</body></html>'
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create full path
    pdf_path = os.path.join(output_folder, filename)
    
    # Generate PDF
    try:
        with open(pdf_path, 'wb') as f:
            pisa_status = pisa.CreatePDF(html, dest=f)
        
        if pisa_status.err:
            raise Exception(f"Error generating PDF: {pisa_status.err}")
        
        print(f"PDF successfully generated: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")
        raise Exception(f"Failed to generate PDF: {str(e)}")
