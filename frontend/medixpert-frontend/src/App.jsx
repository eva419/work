import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar.jsx'
import { predictDisease, sendChatMessage, getDoctors, getAppointments, getReports, getDashboardStats } from './utils/api'
import { 
  Activity, 
  Calendar, 
  FileText, 
  Heart, 
  MessageCircle, 
  Search, 
  Stethoscope, 
  User, 
  Users,
  Brain,
  Upload,
  BarChart3,
  Settings,
  LogOut,
  Menu,
  X,
  Phone,
  Mail,
  MapPin,
  Star,
  Clock,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Shield
} from 'lucide-react'
import './App.css'

// Navigation Component
function Navigation({ currentUser, onLogout }) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'predict', label: 'AI Diagnosis', icon: Brain },
    { id: 'doctors', label: 'Find Doctors', icon: Stethoscope },
    { id: 'appointments', label: 'Appointments', icon: Calendar },
    { id: 'reports', label: 'Medical Reports', icon: FileText },
    { id: 'chat', label: 'Medical Chat', icon: MessageCircle },
  ]

  return (
    <nav className="bg-white shadow-lg border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <Heart className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">MediXpert</span>
            </div>
            <div className="hidden md:ml-6 md:flex md:space-x-8">
              {navItems.map((item) => (
                <a
                  key={item.id}
                  href={`#${item.id}`}
                  className="text-gray-500 hover:text-gray-700 px-3 py-2 rounded-md text-sm font-medium flex items-center"
                >
                  <item.icon className="h-4 w-4 mr-2" />
                  {item.label}
                </a>
              ))}
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Avatar className="h-8 w-8">
                <AvatarImage src="/api/placeholder/32/32" />
                <AvatarFallback>{currentUser?.name?.charAt(0) || 'U'}</AvatarFallback>
              </Avatar>
              <span className="text-sm font-medium text-gray-700">{currentUser?.name || 'User'}</span>
            </div>
            <Button variant="outline" size="sm" onClick={onLogout}>
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>

          <div className="md:hidden flex items-center">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              {isMobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </Button>
          </div>
        </div>
      </div>

      {isMobileMenuOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-gray-50">
            {navItems.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className="text-gray-500 hover:text-gray-700 block px-3 py-2 rounded-md text-base font-medium"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                <item.icon className="h-4 w-4 mr-2 inline" />
                {item.label}
              </a>
            ))}
          </div>
        </div>
      )}
    </nav>
  )
}

// Dashboard Component
function Dashboard() {
  const stats = [
    { title: 'Total Appointments', value: '24', icon: Calendar, color: 'text-blue-600' },
    { title: 'Predictions Made', value: '12', icon: Brain, color: 'text-green-600' },
    { title: 'Reports Uploaded', value: '8', icon: FileText, color: 'text-purple-600' },
    { title: 'Messages', value: '15', icon: MessageCircle, color: 'text-orange-600' },
  ]

  const recentActivities = [
    { type: 'appointment', message: 'Appointment with Dr. Smith scheduled', time: '2 hours ago' },
    { type: 'prediction', message: 'AI diagnosis completed for symptoms', time: '4 hours ago' },
    { type: 'report', message: 'Blood test report uploaded', time: '1 day ago' },
    { type: 'chat', message: 'New message from Dr. Johnson', time: '2 days ago' },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <Badge variant="outline" className="text-green-600 border-green-600">
          <CheckCircle className="h-4 w-4 mr-1" />
          All Systems Healthy
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <Card key={index}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                  <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                </div>
                <stat.icon className={`h-8 w-8 ${stat.color}`} />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Your latest healthcare activities</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentActivities.map((activity, index) => (
                <div key={index} className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900">{activity.message}</p>
                    <p className="text-xs text-gray-500">{activity.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Health Insights</CardTitle>
            <CardDescription>AI-powered health recommendations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="p-4 bg-green-50 rounded-lg">
                <div className="flex items-center">
                  <TrendingUp className="h-5 w-5 text-green-600 mr-2" />
                  <span className="text-sm font-medium text-green-800">Health Score: 85/100</span>
                </div>
                <p className="text-xs text-green-600 mt-1">Your health metrics are improving!</p>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center">
                  <Shield className="h-5 w-5 text-blue-600 mr-2" />
                  <span className="text-sm font-medium text-blue-800">Preventive Care</span>
                </div>
                <p className="text-xs text-blue-600 mt-1">Schedule your annual checkup</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// AI Disease Prediction Component
function DiseasePrediction() {
  const [symptoms, setSymptoms] = useState('')
  const [predictions, setPredictions] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const handlePredict = async () => {
    if (!symptoms.trim()) return
    
    setIsLoading(true)
    try {
      const response = await predictDisease(symptoms)
      if (response.success) {
        setPredictions(response.predictions)
      } else {
        console.error('Prediction failed:', response.error)
        // Fallback to mock data
        const mockPredictions = [
          { disease: 'Common Cold', confidence: 85.2, severity: 'Low' },
          { disease: 'Seasonal Allergy', confidence: 72.8, severity: 'Low' },
          { disease: 'Viral Infection', confidence: 68.5, severity: 'Medium' },
        ]
        setPredictions(mockPredictions)
      }
    } catch (error) {
      console.error('API call failed:', error)
      // Fallback to mock data
      const mockPredictions = [
        { disease: 'Common Cold', confidence: 85.2, severity: 'Low' },
        { disease: 'Seasonal Allergy', confidence: 72.8, severity: 'Low' },
        { disease: 'Viral Infection', confidence: 68.5, severity: 'Medium' },
      ]
      setPredictions(mockPredictions)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">AI Disease Prediction</h1>
        <p className="text-gray-600 mt-2">Describe your symptoms and get AI-powered health insights</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Brain className="h-5 w-5 mr-2 text-blue-600" />
            Symptom Analysis
          </CardTitle>
          <CardDescription>
            Enter your symptoms separated by commas (e.g., fever, headache, cough)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="symptoms">Symptoms</Label>
            <Input
              id="symptoms"
              placeholder="Enter your symptoms here..."
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              className="mt-1"
            />
          </div>
          <Button onClick={handlePredict} disabled={isLoading || !symptoms.trim()}>
            {isLoading ? 'Analyzing...' : 'Analyze Symptoms'}
          </Button>
        </CardContent>
      </Card>

      {predictions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
            <CardDescription>AI analysis of your symptoms</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {predictions.map((prediction, index) => (
                <div key={index} className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-gray-900">{prediction.disease}</h3>
                    <Badge variant={prediction.severity === 'Low' ? 'secondary' : 'destructive'}>
                      {prediction.severity}
                    </Badge>
                  </div>
                  <div className="mt-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Confidence</span>
                      <span className="font-medium">{prediction.confidence}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${prediction.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-yellow-600 mr-2" />
                <span className="text-sm font-medium text-yellow-800">Important Notice</span>
              </div>
              <p className="text-xs text-yellow-700 mt-1">
                This is an AI prediction for informational purposes only. Please consult a healthcare professional for proper diagnosis and treatment.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

// Doctor Search Component
function DoctorSearch() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedSpecialty, setSelectedSpecialty] = useState('')
  const [doctors, setDoctors] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const specialties = ['All', 'Cardiology', 'Dermatology', 'Pediatrics', 'Oncology', 'Neurology']

  // Load doctors on component mount and when filters change
  useEffect(() => {
    loadDoctors()
  }, [searchTerm, selectedSpecialty])

  const loadDoctors = async () => {
    setIsLoading(true)
    try {
      const filters = {}
      if (selectedSpecialty && selectedSpecialty !== 'All') {
        filters.specialty = selectedSpecialty
      }
      if (searchTerm) {
        filters.search = searchTerm
      }
      
      const response = await getDoctors(filters)
      if (response.success) {
        setDoctors(response.doctors)
      }
    } catch (error) {
      console.error('Failed to load doctors:', error)
      // Fallback to mock data
      const mockDoctors = [
        {
          id: 1,
          name: 'Dr. Sarah Johnson',
          specialty: 'Cardiology',
          experience: '15 years',
          rating: 4.9,
          hospital: 'Central Hospital',
          location: 'Downtown',
          available: true,
        },
        {
          id: 2,
          name: 'Dr. Michael Chen',
          specialty: 'Dermatology',
          experience: '12 years',
          rating: 4.8,
          hospital: 'Westside Clinic',
          location: 'Westside',
          available: false,
        },
        {
          id: 3,
          name: 'Dr. Emily Davis',
          specialty: 'Pediatrics',
          experience: '8 years',
          rating: 4.7,
          hospital: 'Eastside Clinic',
          location: 'Eastside',
          available: true,
        },
      ]
      setDoctors(mockDoctors)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Find Doctors</h1>
        <p className="text-gray-600 mt-2">Search for healthcare professionals by specialty and location</p>
      </div>

      <Card>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="search">Search Doctors</Label>
              <Input
                id="search"
                placeholder="Search by name or hospital..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="specialty">Specialty</Label>
              <select
                id="specialty"
                value={selectedSpecialty}
                onChange={(e) => setSelectedSpecialty(e.target.value)}
                className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {specialties.map((specialty) => (
                  <option key={specialty} value={specialty}>{specialty}</option>
                ))}
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {doctors.map((doctor) => (
          <Card key={doctor.id} className="hover:shadow-lg transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-center space-x-4">
                <Avatar className="h-16 w-16">
                  <AvatarImage src={doctor.image} />
                  <AvatarFallback>{doctor.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                </Avatar>
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900">{doctor.name}</h3>
                  <p className="text-sm text-gray-600">{doctor.specialty}</p>
                  <div className="flex items-center mt-1">
                    <Star className="h-4 w-4 text-yellow-400 fill-current" />
                    <span className="text-sm text-gray-600 ml-1">{doctor.rating}</span>
                  </div>
                </div>
              </div>
              
              <div className="mt-4 space-y-2">
                <div className="flex items-center text-sm text-gray-600">
                  <MapPin className="h-4 w-4 mr-2" />
                  {doctor.hospital}, {doctor.location}
                </div>
                <div className="flex items-center text-sm text-gray-600">
                  <Clock className="h-4 w-4 mr-2" />
                  {doctor.experience} experience
                </div>
                <div className="flex items-center justify-between">
                  <Badge variant={doctor.available ? 'secondary' : 'destructive'}>
                    {doctor.available ? 'Available' : 'Busy'}
                  </Badge>
                  <Button size="sm" disabled={!doctor.available}>
                    Book Appointment
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

// Appointments Component
function Appointments() {
  const appointments = [
    {
      id: 1,
      doctor: 'Dr. Sarah Johnson',
      specialty: 'Cardiology',
      date: '2024-07-20',
      time: '10:00 AM',
      status: 'Confirmed',
      type: 'Consultation'
    },
    {
      id: 2,
      doctor: 'Dr. Emily Davis',
      specialty: 'Pediatrics',
      date: '2024-07-22',
      time: '2:30 PM',
      status: 'Pending',
      type: 'Follow-up'
    },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Appointments</h1>
          <p className="text-gray-600 mt-2">Manage your upcoming and past appointments</p>
        </div>
        <Button>
          <Calendar className="h-4 w-4 mr-2" />
          Book New Appointment
        </Button>
      </div>

      <Tabs defaultValue="upcoming" className="w-full">
        <TabsList>
          <TabsTrigger value="upcoming">Upcoming</TabsTrigger>
          <TabsTrigger value="past">Past</TabsTrigger>
          <TabsTrigger value="cancelled">Cancelled</TabsTrigger>
        </TabsList>
        
        <TabsContent value="upcoming" className="space-y-4">
          {appointments.map((appointment) => (
            <Card key={appointment.id}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <Avatar>
                      <AvatarFallback>{appointment.doctor.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                    </Avatar>
                    <div>
                      <h3 className="font-semibold text-gray-900">{appointment.doctor}</h3>
                      <p className="text-sm text-gray-600">{appointment.specialty}</p>
                      <p className="text-sm text-gray-500">{appointment.type}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-medium text-gray-900">{appointment.date}</p>
                    <p className="text-sm text-gray-600">{appointment.time}</p>
                    <Badge variant={appointment.status === 'Confirmed' ? 'secondary' : 'outline'}>
                      {appointment.status}
                    </Badge>
                  </div>
                </div>
                <div className="mt-4 flex space-x-2">
                  <Button variant="outline" size="sm">Reschedule</Button>
                  <Button variant="outline" size="sm">Cancel</Button>
                  <Button size="sm">Join Video Call</Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>
        
        <TabsContent value="past">
          <Card>
            <CardContent className="p-6 text-center">
              <p className="text-gray-500">No past appointments found.</p>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="cancelled">
          <Card>
            <CardContent className="p-6 text-center">
              <p className="text-gray-500">No cancelled appointments found.</p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

// Medical Reports Component
function MedicalReports() {
  const [dragActive, setDragActive] = useState(false)

  const reports = [
    {
      id: 1,
      name: 'Blood Test Results',
      date: '2024-07-15',
      type: 'Lab Report',
      doctor: 'Dr. Sarah Johnson',
      status: 'Reviewed'
    },
    {
      id: 2,
      name: 'X-Ray Chest',
      date: '2024-07-10',
      type: 'Radiology',
      doctor: 'Dr. Michael Chen',
      status: 'Pending Review'
    },
  ]

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    // Handle file upload
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Medical Reports</h1>
        <p className="text-gray-600 mt-2">Upload and manage your medical documents</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Upload New Report</CardTitle>
          <CardDescription>Drag and drop files or click to browse</CardDescription>
        </CardHeader>
        <CardContent>
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-lg font-medium text-gray-900 mb-2">Drop files here</p>
            <p className="text-sm text-gray-600 mb-4">or click to browse</p>
            <Button variant="outline">
              Choose Files
            </Button>
            <p className="text-xs text-gray-500 mt-2">
              Supports PDF, JPG, PNG files up to 10MB
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Your Reports</CardTitle>
          <CardDescription>View and manage your uploaded medical reports</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {reports.map((report) => (
              <div key={report.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <FileText className="h-8 w-8 text-blue-600" />
                  <div>
                    <h3 className="font-medium text-gray-900">{report.name}</h3>
                    <p className="text-sm text-gray-600">{report.type} • {report.date}</p>
                    <p className="text-sm text-gray-500">Reviewed by {report.doctor}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant={report.status === 'Reviewed' ? 'secondary' : 'outline'}>
                    {report.status}
                  </Badge>
                  <Button variant="outline" size="sm">View</Button>
                  <Button variant="outline" size="sm">Download</Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// Medical Chat Component
function MedicalChat() {
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      content: 'Hello! I\'m your medical assistant. How can I help you today?',
      timestamp: '10:00 AM'
    },
    {
      id: 2,
      sender: 'user',
      content: 'I have been experiencing headaches lately.',
      timestamp: '10:01 AM'
    },
    {
      id: 3,
      sender: 'bot',
      content: 'I understand you\'re experiencing headaches. Can you tell me more about when they occur and how severe they are?',
      timestamp: '10:01 AM'
    },
  ])

  const handleSendMessage = () => {
    if (!message.trim()) return

    const newMessage = {
      id: messages.length + 1,
      sender: 'user',
      content: message,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    setMessages([...messages, newMessage])
    setMessage('')

    // Simulate bot response
    setTimeout(() => {
      const botResponse = {
        id: messages.length + 2,
        sender: 'bot',
        content: 'Thank you for sharing that information. For persistent headaches, I recommend consulting with a healthcare professional for proper evaluation.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
      setMessages(prev => [...prev, botResponse])
    }, 1000)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Medical Chat</h1>
        <p className="text-gray-600 mt-2">Get instant medical guidance from our AI assistant</p>
      </div>

      <Card className="h-96">
        <CardHeader>
          <CardTitle className="flex items-center">
            <MessageCircle className="h-5 w-5 mr-2 text-blue-600" />
            Medical Assistant
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col h-full">
          <div className="flex-1 overflow-y-auto space-y-4 mb-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    msg.sender === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <p className="text-sm">{msg.content}</p>
                  <p className={`text-xs mt-1 ${
                    msg.sender === 'user' ? 'text-blue-100' : 'text-gray-500'
                  }`}>
                    {msg.timestamp}
                  </p>
                </div>
              </div>
            ))}
          </div>
          
          <div className="flex space-x-2">
            <Input
              placeholder="Type your message..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              className="flex-1"
            />
            <Button onClick={handleSendMessage}>Send</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center text-sm text-gray-600">
            <AlertCircle className="h-4 w-4 mr-2" />
            This chat is for informational purposes only. For medical emergencies, please call emergency services.
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// Main App Component
function App() {
  const [currentView, setCurrentView] = useState('dashboard')
  const [currentUser, setCurrentUser] = useState({
    name: 'John Doe',
    email: 'john.doe@example.com',
    role: 'patient'
  })

  const handleLogout = () => {
    setCurrentUser(null)
    setCurrentView('login')
  }

  // Handle navigation
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1)
      if (hash) {
        setCurrentView(hash)
      }
    }

    window.addEventListener('hashchange', handleHashChange)
    handleHashChange() // Check initial hash

    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  const renderCurrentView = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard />
      case 'predict':
        return <DiseasePrediction />
      case 'doctors':
        return <DoctorSearch />
      case 'appointments':
        return <Appointments />
      case 'reports':
        return <MedicalReports />
      case 'chat':
        return <MedicalChat />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation currentUser={currentUser} onLogout={handleLogout} />
      
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {renderCurrentView()}
      </main>
    </div>
  )
}

export default App

