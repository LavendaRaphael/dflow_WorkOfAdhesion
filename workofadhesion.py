from dflow.plugins.dispatcher import DispatcherExecutor
from dflow.python import OP
from dflow.python import OPIO
from dflow.python import OPIOSign
from dflow.python import Artifact
from dflow.python import Slices
from dflow.python import PythonOPTemplate
from dflow import Step
from dflow import Workflow
from dflow import download_artifact
from dflow import upload_artifact
from pathlib import Path
import os
import subprocess
import time
from ase.calculators.vasp import Vasp
import ase.io
import copy
import math
import json
from dflow import config
config["mode"] = "debug"

def executor_torque(
    str_jobname: str,
    str_queue: str = 'spst-sunzhr',
    int_nodes: int = 1,
    int_ppn: int = None,
    list_source: list = None,
) -> DispatcherExecutor:

    dict_queue = {
        "spst-sunzhr": 32,
        "spst_pub": 28,
    }

    if int_ppn is None:
        int_ppn = dict_queue[str_queue]

    excutor_torque = DispatcherExecutor(
        host='10.15.22.167',
        username="tianff",
        port = 22112,
        private_key_file = '/home/faye/.ssh/id_rsa',
        queue_name="queue",
        machine_dict = {
            "batch_type":   "Torque",
            "context_type": "SSHContext",
            "remote_root":  "/public/spst/home/tianff/dflow",
        },

        resources_dict = {
            "custom_flags":[
                "#PBS -l walltime=240:00:00",
                f"#PBS -N {str_jobname}"
            ],
            "cpu_per_node": int_ppn,
            "queue_name": str_queue,

            "number_node": int_nodes,
            "group_size": 1,
            "source_list": [
                "$HOME/.config/.tianff",
                "$homedir/.local/bin/bashrc.sh"
            ]
        }
    )

    if not(list_source is None):
        excutor_torque.resources_dict['source_list'].extend(list_source)

    return excutor_torque

class VASPCal(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'path_POSCAR': Artifact(Path),
            'str_outdir': str,
            'ase_vasp': Vasp,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "path_out": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:

        ase_atoms = ase.io.read(
            filename = op_in['path_POSCAR'],
            format = 'vasp'
        )

        ase_atoms.calc = op_in['ase_vasp']
        ase_atoms.calc.set(
            directory = op_in['str_outdir'],
        )

        float_energy = ase_atoms.get_potential_energy()
       
        return OPIO({
            "path_out": Path(op_in['str_outdir']),
        })

class Workofadhesion(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'path_vaspout_a': Artifact(Path),
            'path_vaspout_b': Artifact(Path),
            'path_vaspout_a_b': Artifact(Path),
            'str_json_save': str,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "path_json_save": Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(
        self, 
        op_in: OPIO
    ) -> OPIO:

        ase_atoms_a = ase.io.read(
            filename = op_in['path_vaspout_a']/'OUTCAR',
            format = 'vasp-out'
        )
        ase_atoms_b = ase.io.read(
            filename = op_in['path_vaspout_b']/'OUTCAR',
            format = 'vasp-out'
        )
        ase_atoms_a_b = ase.io.read(
            filename = op_in['path_vaspout_a_b']/'OUTCAR',
            format = 'vasp-out'
        )

        float_energy_a = ase_atoms_a.get_potential_energy()
        float_energy_b = ase_atoms_b.get_potential_energy()
        float_energy_a_b = ase_atoms_a_b.get_potential_energy()

        list_cell = ase_atoms_a_b.cell.cellpar()
        float_area_a_b = list_cell[0] * list_cell[1] * math.sin(math.radians(list_cell[5]))

        float_workofadhesion = (- float_energy_a_b + float_energy_a + float_energy_b)/float_area_a_b

        dict_save = {
            "float_energy_a": float_energy_a,
            "float_energy_b": float_energy_b,
            "float_energy_a_b": float_energy_a_b,
            "float_area_a_b": float_area_a_b,
            "float_workofadhesion": float_workofadhesion,
        }

        with open(op_in['str_json_save'], 'w') as fp:
            json.dump(dict_save, fp, indent=4)

        return OPIO({
            "path_json_save": Path(op_in['str_json_save'])
        })

def main() -> Workflow:

    ase_vasp = Vasp(
        command = 'mpirun vasp_std',
        pp = 'PBE',
        setups = 'recommended',
        npar =  2,
        prec = 'Accurate',
        kspacing = 0.25,
        xc = 'PBE',
        encut = 500,
        ediff = 1e-5,
        nelm = 500,
        lreal = 'Auto',
        ibrion = 2,
        nsw = 500,
        ediffg = -0.01,
        lwave = False,
        lcharg = False
    )

    step_vasp_a = Step(
        "Step-Vasp-Li",
        PythonOPTemplate(
            VASPCal,
            image="my-image"
        ),
        artifacts={
            "path_POSCAR": upload_artifact(['Li.001.x3y5z4_vac12.POSCAR'])
        },
        parameters = {
            'str_outdir': 'Li.001.x3y5z4_vac12',
            'ase_vasp': ase_vasp
        },
        executor=executor_torque(
            str_jobname = 'Li',
            list_source = ['$homedir/.local/bin/bashrc_vasp.6.3.2.sh']
        ),
    )

    ase_vasp_gaussian = copy.deepcopy(ase_vasp)
    ase_vasp_gaussian.set(
        ismear = 0,
        sigma = 0.05,
    )

    step_vasp_b = Step(
        "Step-Vasp-Li2CO3",
        PythonOPTemplate(
            VASPCal,
            image="my-image"
        ),
        artifacts={
            "path_POSCAR": upload_artifact(['Li2CO3.001.x2y2z2_vac12.POSCAR'])
        },
        parameters = {
            'str_outdir': 'Li2CO3.001.x2y2z2_vac12',
            'ase_vasp': ase_vasp_gaussian
        },
        executor=executor_torque(
            str_jobname = 'Li2CO3',
            list_source = ['$homedir/.local/bin/bashrc_vasp.6.3.2.sh']
        ),
    )

    step_vasp_a_b = Step(
        "Step-Vasp-Li-Li2CO3",
        PythonOPTemplate(
            VASPCal,
            image="my-image"
        ),
        artifacts={
            "path_POSCAR": upload_artifact(['Li.001.x5y3z4_Li2CO3.001.x2y2z2_vac12.POSCAR'])
        },
        parameters = {
            'str_outdir': 'Li.001.x5y3z4_Li2CO3.001.x2y2z2_vac12',
            'ase_vasp': ase_vasp_gaussian
        },
        executor=executor_torque(
            str_jobname = 'Li-Li2CO3',
            list_source = ['$homedir/.local/bin/bashrc_vasp.6.3.2.sh']
        ),
    )

    step_workofadhesion = Step(
        "Step-Workofadhesion",
        PythonOPTemplate(
            Workofadhesion,
            image="my-image"
        ),
        artifacts={
            "path_vaspout_a": step_vasp_a.outputs.artifacts['path_out'],
            "path_vaspout_b": step_vasp_b.outputs.artifacts['path_out'],
            "path_vaspout_a_b": step_vasp_a_b.outputs.artifacts['path_out'],
        },
        parameters = {
            'str_json_save': 'workofadhesion.json',
        },
    )

    wf = Workflow("wf-workofadhesion")
    wf.add([step_vasp_a, step_vasp_b, step_vasp_a_b])
    wf.add([step_workofadhesion])
    wf.submit()
    
    while wf.query_status() in ["Pending", "Running"]:
        time.sleep(10)
    assert(wf.query_status() == "Succeeded")

    list_step = ["Step-Vasp-Li", "Step-Vasp-Li2CO3", "Step-Vasp-Li-Li2CO3"]
    for str_step in list_step:
        step = wf.query_step(name=str_step)[0]
        assert(step.phase == "Succeeded")
        download_artifact(step.outputs.artifacts["path_out"])

    step = wf.query_step(name='Step-Workofadhesion')[0]
    assert(step.phase == "Succeeded")
    download_artifact(step.outputs.artifacts["path_json_save"])

if __name__ == "__main__":
    main()
