require 'torch'
require 'MulPos'
require 'NegExp'
require 'RBF'

local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local nntest = {}
local nntestx = {}

function nntest.NegExp()
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()

   local module = nn.NegExp()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.MulPos()      
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ink = math.random(10,20)
   local input = torch.Tensor(ini,inj,ink):zero()
   
   local module = nn.MulPos(ini*inj*ink)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')
   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      print(err)
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.RBF()
   --local ini = math.random(50,70)
   --local inj = math.random(50,70)
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local input = torch.Tensor(ini):zero()
   local module = nn.RBF(ini,inj)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end 


mytester:add(nntest)

   require 'nn'
   jac = nn.Jacobian
   mytester:run()
