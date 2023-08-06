import torch 
import numpy 
import time 
import random 
import chess
import json 
import os 
import sys 



def fen_to_1d(fen:str,dtype=numpy.float32):
		#Encoding will be a bx6x8x8  tensor 
		board_tensor 	= numpy.zeros(shape=(6,8,8),dtype=dtype)
		piece_indx 	= {"R":4,"N":2,"B":3,"Q":5,"K":6,"P":1,"r":-4,"n":-2,"b":-3,"q":-5,"k":-6,"p":-1}
		#Go through FEN and fill pieces
		
		for i in range(1,9):
			fen 	= fen.replace(str(i),"e"*i)

		position	= fen.split(" ")[0].split("/")
		turn 		= fen.split(" ")[1]
		castling 	= fen.split(" ")[2]
		
		#Place pieces
		for rank_i,rank in enumerate(reversed(position)):
			for file_i,piece in enumerate(rank): 
				if not piece == "e":
					board_tensor[0,rank_i,file_i]	= piece_indx[piece]  
		
		#Place turn 
		slice 		= 1 
		board_tensor[slice,:,:]   = numpy.ones(shape=(1,8,8),dtype=dtype) * 1 if turn == "w" else -1

		#Place all castling allows 
		for castle in ["K","Q","k","q"]:
			slice += 1
			board_tensor[slice,:,:]	= numpy.ones(shape=(1,8,8),dtype=dtype) * 1 if castle in castling else 0

		return board_tensor


class TwoPEnv:

	def __init__(self,max_moves,move_space):

		self.turn       = 1 
		self.board      = None
		self.max_moves	= None 
		self.result 	= None 
		self.move_space = move_space
		self.move 		= 0 

	def get_legal_moves(self):

		raise NotImplementedError(f"Implement get_legal_moves in the child class!")

	def make_move(self):

		raise NotImplementedError(f"Implement make_move in the child class!")
	
	def is_game_over(self):
		raise NotImplementedError(f"Implement is_game_over in the child class!")
	
	def get_uuid(self):
		raise NotImplementedError(f"Implement get_uuid in the child class!")
	
	def get_repr(self):
		raise NotImplementedError(f"Implement get_repr in the child class!")
	
	def get_result(self):
		raise NotImplementedError(f"Implement get_result in the child class!")

	def build_as_network(self,id):
		raise NotImplementedError(f"Implement build_as_network in the child class!")
	
class Connect4(TwoPEnv):

	def __init__(self,max_moves):

		super(Connect4,self).__init__(max_moves,8)

		#Game board size will be 6 x 8 
		self.nrows      = 6 
		self.ncols      = 8 
		self.board      = numpy.zeros(shape=(self.ncols,self.nrows),dtype=numpy.int8)
		self.turn       = 1
		self.move       = 0 
		self.max_moves  = self.nrows * self.ncols  
		self.result     = None 
		self.last_move  = (None,None)


		self.x_indices  = []
		self.o_indices  = [] 

	#CAN BE OPTIMIZED
	def get_legal_moves(self):

		legal_moves                 = []

		for col in range(self.ncols):
			curcol                  = self.board[col,]
			
			if not numpy.count_nonzero(curcol) == self.nrows: 
				legal_moves.append(col)

		return legal_moves
	

	def make_move(self,col):
		if col < 0:
			raise ValueError(f"NO VALID MOVE FOR {col} in\n{self.board}")    
		for row in range(self.nrows):

			if self.board[col,row] == 0:

				self.x_indices.append((col,row)) if self.turn else self.o_indices.append((col,row)) 

				self.board[col,row]     = self.turn 
				self.turn               *= -1 
				self.last_move          = (col,row)
				self.move               += 1 
				

				return 

		raise ValueError(f"NO VALID MOVE FOR {col} in\n{self.board}")
	

	#ASSUMES ALL MOVES UP UNTIL NOW WERE NOT GAME OVER 
	def is_game_over(self):
		
		if self.move == 0:
			return None
		if self.move == self.max_moves:
			return 0
		start_col,start_row     = self.last_move

		potential_winner    = self.turn * -1
		
		#print(f"last move was {self.last_move}")
		#Check Horiz + vert
		for presumed_start_offset in [-3,-2,-1,0]:
			x1,x2,x3,x4 = start_col+presumed_start_offset,start_col+(presumed_start_offset+1),start_col+(presumed_start_offset+2),start_col+(presumed_start_offset+3)
			#print(f"\nHORIZ\nind = {[x1,x2,x3,x4]}")

			#Disregard if not len 4:
			segment_horiz     = self.board[x1:x4+1,start_row]
			#print(f"segment_horiz is {segment_horiz}")
			if len(segment_horiz) == 4 and numpy.all(segment_horiz == potential_winner): #Since move just got flipped
				self.result = potential_winner
				return potential_winner 
		
		#for presumed_start_offset in [-3,-2,-1,0]:
			y1,y2,y3,y4 = start_row+presumed_start_offset,start_row+(presumed_start_offset+1),start_row+(presumed_start_offset+2),start_row+(presumed_start_offset+3)
			#print(f"\n\nVERT\nind = {[y1,y2,y3,y4]}")
			#Disregard if not len 4:
			segment_vert     = self.board[start_col,y1:y4+1]
			#print(f"segment_vert is {segment_vert}")
			if len(segment_vert) == 4 and numpy.all(segment_vert == potential_winner): #Since move just got flipped
				self.result = potential_winner
				return potential_winner 
		

		#Check DIAGONAL

		for presumed_start_offset in [-3,-2,-1,0]:
			x1,x2,x3,x4 = start_col+presumed_start_offset,start_col+(presumed_start_offset+1),start_col+(presumed_start_offset+2),start_col+(presumed_start_offset+3)
			
			if x1 >= self.ncols or x2 >= self.ncols or x3 >= self.ncols or x4 >= self.ncols or x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0:
				continue 

			y1,y2,y3,y4 = start_row+presumed_start_offset,start_row+(presumed_start_offset+1),start_row+(presumed_start_offset+2),start_row+(presumed_start_offset+3)
			diagonal    = numpy.zeros(shape=(4),dtype=float) 
			good_diag   = True 

			for i,coord in enumerate(list(zip([x1,x2,x3,x4],[y1,y2,y3,y4]))):
				x,y     = coord 
				if y < 0 or y >= self.nrows:
					good_diag = False
					break
				diagonal[i] = self.board[coord[0],coord[1]]
			
			if good_diag:
				if numpy.all(diagonal == potential_winner):
					self.result = potential_winner
					return potential_winner 
				#print(f"searching diagonal {diagonal}")

		#for presumed_start_offset in [-3,-2,-1,0]:
			y1,y2,y3,y4 = start_row-presumed_start_offset,start_row-(presumed_start_offset+1),start_row-(presumed_start_offset+2),start_row-(presumed_start_offset+3)
			#x1,x2,x3,x4 = start_col+presumed_start_offset,start_col+(presumed_start_offset+1),start_col+(presumed_start_offset+2),start_col+(presumed_start_offset+3)

			
			diagonal    = numpy.zeros(shape=(4),dtype=float) 
			good_diag   = True 
			for i,coord in enumerate(list(zip([x1,x2,x3,x4],[y1,y2,y3,y4]))):
				x,y     = coord 
				if y < 0 or y >= self.nrows:
					good_diag = False
					break
				diagonal[i] = self.board[coord[0],coord[1]]
			
			if good_diag:
				if numpy.all(diagonal == potential_winner):
					self.result = potential_winner
					return potential_winner 
				#print(f"searching diagonal {diagonal}")
		return None

	
	def get_uuid(self):
		return f"{self.x_indices}{self.o_indices}"


	def get_repr(self):
		return torch.from_numpy(self.board).float().to(torch.device('cuda' if torch.cuda.is_available() else "cpu"))


	def get_result(self):
		if self.move == self.ncols*self.nrows:
			return 0
		return self.result


	def build_as_network(self,id):
		addl        = numpy.zeros(shape=(self.nrows))
		addl[0]     = id 

		return numpy.vstack([self.board,addl])
	
	
	def __repr__(self):

		s   = ""

		for row in reversed(range(self.nrows)):
			for col in range(self.ncols):
				marker = "."
				if self.board[col,row] == -1:
					marker = "O"
				elif self.board[col,row] == 1:
					marker  = "X"
				s   += f"|{marker}"
			s+= "|\n"
		
		return s 


class Chess(TwoPEnv):


	def __init__(self,max_moves,tensorizing_fn=fen_to_1d,chess_moves=None,move_to_index=None,id=0,gen=0):

		super(Chess,self).__init__(max_moves,1968)
		self.board      	= chess.Board()
		self.turn       	= self.board.turn
		self.move       	= 0 
		self.max_moves  	= max_moves
		self.result     	= None 
		self.tensorizing	= tensorizing_fn
		self.id 			= id 
		self.gen 			= gen

		if chess_moves is None:
			try:
				self.chess_moves 		= json.loads(open("C:/gitrepos/steinpy/ml/res/chessmoves.txt","r").read())
			except FileNotFoundError:
				self.chess_moves 		= json.loads(open(os.path.join("/home/steinshark/code","steinpy","ml","res","chessmoves.txt"),"r").read())
		else:
			self.chess_moves	= chess_moves
		
		if move_to_index is None:
			self.move_to_index 	= {chess.Move.from_uci(uci):i for i,uci in enumerate(self.chess_moves)}
		else:
			self.move_to_index 	= move_to_index
		

	def get_legal_moves(self):
		return [self.move_to_index[move] for move in self.board.generate_legal_moves()]

	
	def make_move(self,move:int or chess.Move):

		if isinstance(move,chess.Move):
			self.board.push(move)

		elif isinstance(move,int):
			self.board.push_san(self.chess_moves[move])
		
		self.move  += 1 

	
	def is_game_over(self):

		game_over 			=  self.board.is_checkmate() or self.board.is_insufficient_material() or self.board.is_stalemate() or self.board.is_seventyfive_moves() or self.move > self.max_moves

		if game_over:

			res 		= self.board.result()
			if "1" in res and not "1/2" in res:
				if res[0] == "1":
					self.result 	= 1 
					return 1 
				else:
					self.result 	= -1 
					return -1 
			elif "1/2" in res or "*" in res:
				self.result 	=  0
				return 0  
			else:
				print(f"weird result '{res}'")
		
		else:
			return None 
	

	def get_uuid(self):
		return self.board.fen()
	
	
	def get_repr(self):
		return torch.tensor(self.tensorizing(self.board.fen())).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


	def get_result(self):
		return self.result


	def build_as_network(self):

		return (self.tensorizing(self.board.fen(),dtype=numpy.int8),self.id,self.gen)


	def copy(self):
		returning_copy 			= Chess(self.max_moves,tensorizing_fn=self.tensorizing,chess_moves=self.chess_moves,move_to_index=self.move_to_index)
		returning_copy.board 	= self.board.copy(stack=False)
		returning_copy.turn 	= self.turn 
		returning_copy.move 	= self.move 
		returning_copy.result 	= self.result

		return returning_copy



		

if __name__ == "__main__":

	t1 = time.time()

	games_spread    = []
	ngames          = 1000
	for repeat in range(ngames):
		c4  = Connect4()

		while not c4.is_game_over():
			move    = random.choice(c4.get_legal_moves())
			c4.make_move(move)

		#input(str(c4))
		games_spread.append(c4.is_game_over())

	print(f"time to play {ngames} games is {(time.time()-t1):.2f}s white wins {games_spread.count(1)}, black wins {games_spread.count(-1)}, draw happens {games_spread.count(0)}")






	


