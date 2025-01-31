# Campus Recycling Revolution

## Overview
**Campus Recycling Revolution** is a waste management system developed for **Northwest Missouri State University** to streamline waste tracking, recycling revenue, and landfill expenses. The system enables user authentication, role-based access control, and interactive data visualizations for informed decision-making.

## Features
- **User Authentication & Role-Based Access Control:** Secure login system with admin and user roles.
- **Waste Logging System:** Users can log waste data categorized by type and date.
- **Dynamic Category Management:** Admins can add/delete waste categories and subcategories.
- **Automated Reports:** Generates reports for:
  - Recycling revenue
  - Landfill expenses
  - Waste diversion rate
  - Monthly waste summaries
- **Data Visualization:** Uses **Matplotlib** to generate:
  - **Pie Charts** (waste distribution)
  - **Bar Graphs** (monthly waste trends)
  - **Summary Tables**
- **Excel Export:** Users can download reports in **Excel** format for offline analysis.

## Technologies Used
- **Backend:** Flask, SQLAlchemy, SQLite
- **Frontend:** HTML, CSS, Bootstrap, JavaScript
- **Security:** Bcrypt (password hashing)
- **Data Processing:** Pandas, Matplotlib, ExcelWriter
- **Database Models:**
  - `User`: Manages authentication & roles
  - `WasteRecord`: Logs daily waste collection data
  - `Category`: Organizes waste types & subcategories
  - `RecyclingRevenue`: Tracks recycling sales & earnings
  - `LandfillExpense`: Monitors landfill costs & weight

## Installation & Setup
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-repo/campus-recycling-revolution.git
   cd campus-recycling-revolution
   ## Installation & Setup

### Create a Virtual Environment & Install Dependencies:
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Set Up the Database:
python database.py

Run the Application:
python app.py

Open http://127.0.0.1:5000/ in your browser.

## Usage

### Admin:
- Logs in as **admin** (default username: `admin`, password: `adminpassword`)
- Manages users, waste categories, reports
- Adds recycling revenue and landfill expenses

### Users:
- Logs waste data daily
- Views analytics and reports

## Future Enhancements
- **Multi-User Analytics:** Graphs comparing different users’ waste management efforts.
- **Machine Learning Forecasting:** Predicts waste trends for better planning.
- **Cloud Database Integration:** Upgrade from SQLite to PostgreSQL for scalability.

   
   
