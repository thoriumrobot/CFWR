public class PreAndPostDec {

    void pre1(int[] args) {
        int ii = 0;
        while ((ii < args.Length)) {
            // :: error: (array.access.unsafe.high)
            int m = args[++ii];
        }
    }

    void pre2(int[] args) {
        int ii = 0;
        while ((ii < args.Length)) {
            ii++;
            // :: error: (array.access.unsafe.high)
            int m = args[ii];
        }
    }

    void post1(int[] args) {
        int ii = 0;
        while ((ii < args.Length)) {
            int m = args[ii++];
        }
    }

    void post2(int[] args) {
        int ii = 0;
        while ((ii < args.Length)) {
            int m = args[ii];
            ii++;
        }
    }
}
