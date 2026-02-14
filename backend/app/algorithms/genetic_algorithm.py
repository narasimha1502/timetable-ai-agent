"""
Genetic Algorithm for Timetable Scheduling
==========================================

WHY GENETIC ALGORITHM?
- Timetable scheduling is an NP-hard problem
- GA efficiently explores huge solution spaces
- Mimics natural evolution: survival of the fittest

HOW IT WORKS:
1. Create random population of timetables (chromosomes)
2. Evaluate fitness (how good is each timetable?)
3. Select best ones as parents
4. Create offspring through crossover (combine parents)
5. Mutate randomly (explore new solutions)
6. Repeat until optimal solution found
"""

import random
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from copy import deepcopy


# ============= DATA STRUCTURES =============

@dataclass
class TimeSlot:
    """Represents a time slot in the timetable"""
    day: str  # "Monday", "Tuesday", etc.
    period: int  # 1, 2, 3... (e.g., 1 = 9-10 AM)
    
    def __hash__(self):
        return hash((self.day, self.period))
    
    def __eq__(self, other):
        return self.day == other.day and self.period == other.period
    
    def __repr__(self):
        return f"{self.day}-P{self.period}"


@dataclass
class Course:
    """Represents a course to be scheduled"""
    id: str
    name: str
    teacher_id: str
    students_count: int
    duration: int = 1  # Number of consecutive periods
    requires_lab: bool = False
    section: str = "A"  # For multiple sections
    
    def __repr__(self):
        return f"{self.id}({self.section})"


@dataclass
class Teacher:
    """Represents a teacher"""
    id: str
    name: str
    preferred_slots: List[TimeSlot] = None
    max_hours_per_day: int = 6
    
    def __post_init__(self):
        if self.preferred_slots is None:
            self.preferred_slots = []


@dataclass
class Room:
    """Represents a classroom"""
    id: str
    capacity: int
    is_lab: bool = False
    
    def __repr__(self):
        return f"{self.id}({'Lab' if self.is_lab else 'Room'})"


@dataclass
class Gene:
    """
    One gene = one course assignment
    Represents: "Course X is taught in Room Y at Time Z"
    """
    course: Course
    room: Room
    time_slot: TimeSlot
    
    def __repr__(self):
        return f"{self.course.name}|{self.room.id}|{self.time_slot}"


class Chromosome:
    """
    One chromosome = complete timetable
    A list of genes representing all course assignments
    """
    
    def __init__(self, genes: List[Gene]):
        self.genes = genes
        self.fitness: float = 0.0
    
    def __len__(self):
        return len(self.genes)
    
    def copy(self):
        return Chromosome([deepcopy(gene) for gene in self.genes])
    
    def __repr__(self):
        return f"Chromosome(genes={len(self.genes)}, fitness={self.fitness:.2f})"


# ============= GENETIC ALGORITHM ENGINE =============

class TimetableGA:
    """
    Genetic Algorithm for Timetable Optimization
    
    PARAMETERS:
    - population_size: Number of timetables in each generation
    - mutation_rate: Probability of random changes (0.01 = 1%)
    - crossover_rate: Probability of combining parents (0.8 = 80%)
    - elite_size: Number of best individuals that always survive
    """
    
    def __init__(
        self,
        courses: List[Course],
        teachers: List[Teacher],
        rooms: List[Room],
        time_slots: List[TimeSlot],
        population_size: int = 100,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        elite_size: int = 10
    ):
        self.courses = courses
        self.teachers = {t.id: t for t in teachers}
        self.rooms = rooms
        self.time_slots = time_slots
        
        # GA parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # Tracking
        self.best_fitness_history = []
        self.generation = 0
    
    def create_random_chromosome(self) -> Chromosome:
        """
        Create one random timetable
        WHY: Initial population needs diversity to explore solution space
        """
        genes = []
        for course in self.courses:
            # Find valid rooms for this course
            valid_rooms = [
                r for r in self.rooms 
                if (not course.requires_lab or r.is_lab)
                and r.capacity >= course.students_count
            ]
            
            if not valid_rooms:
                # Fallback: use any room if no valid ones
                valid_rooms = self.rooms
            
            # Randomly assign room and time
            room = random.choice(valid_rooms)
            time_slot = random.choice(self.time_slots)
            
            gene = Gene(course=course, room=room, time_slot=time_slot)
            genes.append(gene)
        
        return Chromosome(genes)
    
    def initialize_population(self) -> List[Chromosome]:
        """Create initial population of random timetables"""
        print(f"🧬 Initializing population of {self.population_size} timetables...")
        return [self.create_random_chromosome() for _ in range(self.population_size)]
    
    def calculate_fitness(self, chromosome: Chromosome) -> float:
        """
        Calculate fitness score for a timetable
        
        SCORING LOGIC:
        - Start at 1000 (perfect score)
        - Subtract penalties for violations
        - Hard constraints: -100 points (must satisfy)
        - Soft constraints: -5 to -10 points (desirable)
        
        Higher score = Better timetable
        """
        score = 1000.0
        penalty_details = []
        
        # HARD CONSTRAINT 1: No teacher conflicts
        # Teacher cannot teach two courses at same time
        teacher_schedule = {}
        teacher_conflicts = 0
        for gene in chromosome.genes:
            key = (gene.course.teacher_id, gene.time_slot)
            if key in teacher_schedule:
                teacher_conflicts += 1
                score -= 100
            else:
                teacher_schedule[key] = gene
        
        if teacher_conflicts > 0:
            penalty_details.append(f"Teacher conflicts: {teacher_conflicts}")
        
        # HARD CONSTRAINT 2: No room conflicts
        # Room cannot host two courses simultaneously
        room_schedule = {}
        room_conflicts = 0
        for gene in chromosome.genes:
            key = (gene.room.id, gene.time_slot)
            if key in room_schedule:
                room_conflicts += 1
                score -= 100
            else:
                room_schedule[key] = gene
        
        if room_conflicts > 0:
            penalty_details.append(f"Room conflicts: {room_conflicts}")
        
        # HARD CONSTRAINT 3: Room capacity must be sufficient
        capacity_violations = 0
        for gene in chromosome.genes:
            if gene.course.students_count > gene.room.capacity:
                capacity_violations += 1
                score -= 100
        
        if capacity_violations > 0:
            penalty_details.append(f"Capacity violations: {capacity_violations}")
        
        # HARD CONSTRAINT 4: Lab requirements must be met
        lab_violations = 0
        for gene in chromosome.genes:
            if gene.course.requires_lab and not gene.room.is_lab:
                lab_violations += 1
                score -= 100
        
        if lab_violations > 0:
            penalty_details.append(f"Lab violations: {lab_violations}")
        
        # SOFT CONSTRAINT 1: Teacher preferences
        preference_violations = 0
        for gene in chromosome.genes:
            teacher = self.teachers[gene.course.teacher_id]
            if teacher.preferred_slots and gene.time_slot not in teacher.preferred_slots:
                preference_violations += 1
                score -= 5
        
        # SOFT CONSTRAINT 2: Balanced daily workload
        teacher_daily_hours = {}
        workload_violations = 0
        for gene in chromosome.genes:
            key = (gene.course.teacher_id, gene.time_slot.day)
            teacher_daily_hours[key] = teacher_daily_hours.get(key, 0) + gene.course.duration
        
        for (teacher_id, day), hours in teacher_daily_hours.items():
            teacher = self.teachers[teacher_id]
            if hours > teacher.max_hours_per_day:
                excess = hours - teacher.max_hours_per_day
                workload_violations += excess
                score -= 10 * excess
        
        if workload_violations > 0:
            penalty_details.append(f"Workload violations: {workload_violations}h")
        
        # SOFT CONSTRAINT 3: Minimize gaps in teacher schedules
        gaps = 0
        for teacher_id in self.teachers.keys():
            for day in set(ts.day for ts in self.time_slots):
                periods = sorted([
                    gene.time_slot.period 
                    for gene in chromosome.genes 
                    if gene.course.teacher_id == teacher_id 
                    and gene.time_slot.day == day
                ])
                if len(periods) > 1:
                    gap_count = sum(periods[i+1] - periods[i] - 1 for i in range(len(periods)-1))
                    gaps += gap_count
                    score -= 3 * gap_count
        
        if gaps > 0:
            penalty_details.append(f"Schedule gaps: {gaps}")
        
        # Store penalty details for debugging
        chromosome.penalty_details = penalty_details
        
        return max(0, score)
    
    def tournament_selection(self, population: List[Chromosome], tournament_size: int = 5) -> Chromosome:
        """
        Select parent using tournament selection
        WHY: Balances exploration and exploitation
        - Pick K random individuals
        - Return the best one
        - Better individuals more likely to be selected
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda c: c.fitness)
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Combine two parent timetables
        METHOD: Two-point crossover
        - Cut both parents at two random points
        - Swap middle segments
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        size = len(parent1.genes)
        if size < 2:
            return parent1.copy(), parent2.copy()
        
        point1, point2 = sorted(random.sample(range(size), 2))
        
        child1_genes = (parent1.genes[:point1] + 
                       parent2.genes[point1:point2] + 
                       parent1.genes[point2:])
        child2_genes = (parent2.genes[:point1] + 
                       parent1.genes[point1:point2] + 
                       parent2.genes[point2:])
        
        return Chromosome(child1_genes), Chromosome(child2_genes)
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Randomly modify genes
        WHY: Introduces new variations, prevents premature convergence
        """
        mutated = chromosome.copy()
        
        for gene in mutated.genes:
            if random.random() < self.mutation_rate:
                # 50% chance to change room, 50% to change time
                if random.random() < 0.5:
                    # Change room
                    valid_rooms = [
                        r for r in self.rooms 
                        if (not gene.course.requires_lab or r.is_lab)
                        and r.capacity >= gene.course.students_count
                    ]
                    if valid_rooms:
                        gene.room = random.choice(valid_rooms)
                else:
                    # Change time slot
                    gene.time_slot = random.choice(self.time_slots)
        
        return mutated
    
    def evolve(self, max_generations: int = 1000, target_fitness: float = 950) -> Chromosome:
        """
        Main evolution loop
        
        PROCESS:
        1. Initialize random population
        2. For each generation:
           - Evaluate fitness
           - Select elite (best individuals)
           - Create offspring via selection + crossover
           - Mutate offspring
           - Form new population
        3. Repeat until target reached or max generations
        """
        print(f"\n{'='*60}")
        print(f"  GENETIC ALGORITHM - TIMETABLE OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Courses to schedule: {len(self.courses)}")
        print(f"Available rooms: {len(self.rooms)}")
        print(f"Time slots: {len(self.time_slots)}")
        print(f"Population size: {self.population_size}")
        print(f"Target fitness: {target_fitness}")
        print(f"{'='*60}\n")
        
        # Initialize
        population = self.initialize_population()
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluate fitness for all chromosomes
            for chromosome in population:
                chromosome.fitness = self.calculate_fitness(chromosome)
            
            # Sort by fitness (best first)
            population.sort(key=lambda c: c.fitness, reverse=True)
            
            # Track best solution
            best = population[0]
            self.best_fitness_history.append(best.fitness)
            
            # Progress update every 50 generations
            if generation % 50 == 0 or generation < 10:
                print(f"Generation {generation:4d}: Best Fitness = {best.fitness:7.2f} | "
                      f"Avg = {np.mean([c.fitness for c in population]):7.2f}")
                if hasattr(best, 'penalty_details') and best.penalty_details:
                    print(f"                 Penalties: {', '.join(best.penalty_details[:3])}")
            
            # Check if solution found
            if best.fitness >= target_fitness:
                print(f"\n{'='*60}")
                print(f"✅ SOLUTION FOUND in generation {generation}!")
                print(f"Final fitness: {best.fitness:.2f}")
                print(f"{'='*60}\n")
                return best
            
            # Create new population
            new_population = []
            
            # Elitism: Keep top performers
            new_population.extend([p.copy() for p in population[:self.elite_size]])
            
            # Fill rest with offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to exact size
            population = new_population[:self.population_size]
        
        # Max generations reached
        print(f"\n{'='*60}")
        print(f"⚠️  Reached max generations ({max_generations})")
        print(f"Best fitness achieved: {population[0].fitness:.2f}")
        print(f"{'='*60}\n")
        return population[0]
    
    def chromosome_to_dict(self, chromosome: Chromosome) -> Dict:
        """Convert chromosome to JSON-friendly format"""
        schedule = {}
        for gene in chromosome.genes:
            key = f"{gene.time_slot.day}_{gene.time_slot.period}"
            if key not in schedule:
                schedule[key] = []
            schedule[key].append({
                "course": gene.course.name,
                "course_id": gene.course.id,
                "teacher": gene.course.teacher_id,
                "room": gene.room.id,
                "students": gene.course.students_count,
                "section": gene.course.section,
                "requires_lab": gene.course.requires_lab
            })
        
        return {
            "fitness": chromosome.fitness,
            "generation": self.generation,
            "schedule": schedule
        }


# ============= HELPER FUNCTIONS =============

def create_time_slots(days: List[str], periods_per_day: int) -> List[TimeSlot]:
    """Generate all time slots for a week"""
    slots = []
    for day in days:
        for period in range(1, periods_per_day + 1):
            slots.append(TimeSlot(day=day, period=period))
    return slots


def print_timetable(schedule_dict: Dict, days: List[str], periods_per_day: int):
    """Pretty print a timetable"""
    print("\n" + "="*80)
    print("  GENERATED TIMETABLE")
    print("="*80)
    
    for day in days:
        print(f"\n📅 {day.upper()}")
        print("-"*80)
        for period in range(1, periods_per_day + 1):
            key = f"{day}_{period}"
            if key in schedule_dict["schedule"]:
                classes = schedule_dict["schedule"][key]
                print(f"\n  Period {period}:")
                for cls in classes:
                    lab_mark = " 🔬" if cls['requires_lab'] else ""
                    print(f"    • {cls['course']:30} | {cls['teacher']:8} | "
                          f"Room {cls['room']:6} | {cls['students']} students{lab_mark}")
            else:
                print(f"\n  Period {period}: [Free]")
    
    print("\n" + "="*80)
    print(f"Fitness Score: {schedule_dict['fitness']:.2f} / 1000")
    print(f"Generation: {schedule_dict['generation']}")
    print("="*80 + "\n")


# ============= DEMO / TESTING =============

if __name__ == "__main__":
    print("\n🎓 TIMETABLE AI - GENETIC ALGORITHM DEMO\n")
    
    # Sample data: Small college department
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    time_slots = create_time_slots(days, periods_per_day=8)
    
    teachers = [
        Teacher(id="T001", name="Dr. Smith", max_hours_per_day=6),
        Teacher(id="T002", name="Prof. Johnson", max_hours_per_day=5),
        Teacher(id="T003", name="Dr. Williams", max_hours_per_day=6),
        Teacher(id="T004", name="Prof. Brown", max_hours_per_day=5),
        Teacher(id="T005", name="Dr. Davis", max_hours_per_day=6),
    ]
    
    rooms = [
        Room(id="R101", capacity=50, is_lab=False),
        Room(id="R102", capacity=60, is_lab=False),
        Room(id="R103", capacity=40, is_lab=False),
        Room(id="L201", capacity=30, is_lab=True),
        Room(id="L202", capacity=30, is_lab=True),
    ]
    
    courses = [
        Course("CS101", "Data Structures", "T001", 50, duration=1),
        Course("CS102", "Algorithms", "T001", 45, duration=1),
        Course("CS201", "Database Systems", "T002", 50, duration=1),
        Course("CS202", "Database Lab", "T002", 28, duration=2, requires_lab=True),
        Course("CS301", "Machine Learning", "T003", 40, duration=1),
        Course("CS302", "ML Lab", "T003", 25, duration=2, requires_lab=True),
        Course("CS401", "Web Development", "T004", 55, duration=1),
        Course("CS501", "Artificial Intelligence", "T005", 35, duration=1),
        Course("CS502", "Computer Networks", "T004", 45, duration=1),
        Course("CS601", "Cloud Computing", "T005", 38, duration=1),
    ]
    
    # Run GA
    ga = TimetableGA(
        courses=courses,
        teachers=teachers,
        rooms=rooms,
        time_slots=time_slots,
        population_size=150,
        mutation_rate=0.05,
        crossover_rate=0.8,
        elite_size=15
    )
    
    best_solution = ga.evolve(max_generations=500, target_fitness=950)
    
    # Display result
    result = ga.chromosome_to_dict(best_solution)
    print_timetable(result, days, periods_per_day=8)
    
    print("✅ Demo complete! The algorithm successfully generated a timetable.")
    print("   This code will be integrated into the FastAPI backend next.")
