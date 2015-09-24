#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#include "gammacalculation.h"

namespace msra {
    namespace lattices {
        /*template<class ElemType>
        GammaCalculation::~GammaCalculation()
        {
        }
        template<class ElemType>
        void GammaCalculation::init(msra::asr::simplesenonehmm hset, int DeviceId)
        {
            if (!initialmark)
            {
                m_hset = hset;
                msra::lattices::mbrclassdefinition mbrclassdef = msra::lattices::senone;    // defines the unit for minimum bayesian risk

                // prep for parallel implementation (CUDA)
                if (DeviceId == CPUDEVICE)
                    parallellattice.setmode(true);
                else
                    parallellattice.setmode(false);

                if (parallellattice.enabled())                   // send hmm set to GPU if GPU computation enabled
                    parallellattice.entercomputation(m_hset, mbrclassdef);       // cache senone2classmap if mpemode
                initialmark = true;
            }
        }*/
       /* template<class ElemType>
        void GammaCalculation<ElemType>::calgammaformb(std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>> &lattices, const Microsoft::MSR::CNTK::Matrix<ElemType>& loglikelihood
            , Microsoft::MSR::CNTK::Matrix<float>& outputgamma, const Microsoft::MSR::CNTK::Matrix<ElemType>& label)
        {
            //check total frame number to be added ?
            //int deviceid = loglikelihood.GetDeviceId();

            //convert from Microsoft::MSR::CNTK::Matrix to  msra::math::ssematrixbase
            msra::dbn::matrix pred(loglikelihood.GetNumCols(), loglikelihood.GetNumRows());
            float * p = &pred(0, 0);
            size_t matrixsize = loglikelihood.GetNumCols()*loglikelihood.GetNumRows();

            //loglikelihood.CopyToArray(p, matrixsize);
            msra::dbn::matrix dengammas(loglikelihood.GetNumCols(), loglikelihood.GetNumRows());
            vector<size_t> uids(loglikelihood.GetNumCols());
            vector<size_t> boundary(loglikelihood.GetNumCols());

            msra::dbn::matrix gammasbuffer;

            const float lmf = 14.0f; // Note that 9 was best for Fisher  --these should best be configurable
            const float wp = 0.0f;
            const float amf = 14.0f;
            size_t ts = 0;

            for (size_t i = 0; i < lattices.size(); i++)
            {
                const size_t numframes = lattices[i]->getnumframes();
                msra::dbn::matrixstripe predstripe(pred, ts, numframes);           // logLLs for this utterance
                //msra::dbn::matrixstripe numgammasstripe (numgammas, ts, numframes); // numerator gammas   //[v-hansu] we do not need this currently
                msra::dbn::matrixstripe dengammasstripe(dengammas, ts, numframes); // denominator gammas
                array_ref<size_t> uidsstripe(&uids[ts], numframes);
                //array_ref<size_t> boundsstripe(&boundary[ts], numframes);

                const float boostmmifactor = 0.0f;
                const bool seqsMBRmode = false;
                auto_timer dengammatimer;
                const double denavlogp = lattices[i]->second.forwardbackward(parallellattice,
                    (const msra::math::ssematrixbase &) predstripe, (const msra::asr::simplesenonehmm &) m_hset,
                    (msra::math::ssematrixbase &) dengammasstripe, (msra::math::ssematrixbase &) gammasbuffer
                    lmf, wp, amf, boostmmifactor, seqsMBRmode, uidsstripe, uidsstripe);
                printf("dengamma value %f\n", denavlogp);
                ts += numframes;


            }
        }*/
    }
}