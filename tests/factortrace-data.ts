import {Matrix} from 'ml-matrix';
import {Point} from "../src/index";

export interface TracePoint {
    scale: number;
    projectedUnitVectors: Array<Point>;
}

export class StudentExample {
    public readonly means: Array<Matrix> = [];
    public readonly covs: Array<Matrix> = [];
    public readonly expected2dTracePoints: Array<TracePoint> = [];

    public constructor() {
        this.means.push(new Matrix([[15, 12.285714285714285, 14.095238095238097, 15]]));
        this.means.push(new Matrix([[9, 15.285714285714285, 12.285714285714285, 10]]));
        this.means.push(new Matrix([[6, 10.5, 16.5, 15.285714285714285]]));
        this.means.push(new Matrix([[12.285714285714285, 17.83333315778618, 19, 11]]));
        this.means.push(new Matrix([[2.16666669375, 7.7142857142857135, 12, 14]]));
        this.means.push(new Matrix([[1, 5, 9, 7.5]]));

        this.covs.push(new Matrix([[0.1, 0, 0, 0],
            [0, 1.227891156462585, 0, 0],
            [0, 0, 33.33333333333333, 0],
            [0, 0, 0, 0.3333333333333333]]));
        this.covs.push(new Matrix([[0.1, 0, 0, 0],
            [0, 1.227891156462585, 0, 0],
            [0, 0, 1.227891156462585, 0],
            [0, 0, 0, 0.1]]));
        this.covs.push(new Matrix([[0.1, 0, 0, 0],
            [0, 0.08333333333333333, 0, 0],
            [0, 0, 4.083333333333333, 0],
            [0, 0, 0, 1.227891156462585]]));
        this.covs.push(new Matrix([[1.227891156462585, 0, 0, 0],
            [0, 1.9722222562500011, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.3333333333333333]]));
        this.covs.push(new Matrix([[1.9722221881944448, 0, 0, 0],
            [0, 1.227891156462585, 0, 0],
            [0, 0, 1.3333333333333333, 0],
            [0, 0, 0, 0.1]]));
        this.covs.push(new Matrix([[0.1, 0, 0, 0],
            [0, 0.3333333333333333, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.75]]));

        this.expected2dTracePoints.push({
            scale: 0,
            projectedUnitVectors: [new Point([0.7008923218383587, 0.0060086685798348945]),
                new Point([0.5851299073310892, 0.39198073002162354]),
                new Point([0.3815104578849568, -0.2808835649969672]),
                new Point([0.14430078052164394, -0.8760247862407816])]
        });
        this.expected2dTracePoints.push({
            scale: 0.1,
            projectedUnitVectors: [new Point([0.7007091290604303, 0.007938540933201661]),
                new Point([0.5850561961853812, 0.39170616221643817]),
                new Point([0.38196625715975874, -0.283963837894128]),
                new Point([0.14428354773788535, -0.8751387323264574])]
        });
        this.expected2dTracePoints.push({
            scale: 0.4,
            projectedUnitVectors: [new Point([0.6979056457045778, 0.04228641382084042]),
                new Point([0.5839029658025486, 0.3857633767653745]),
                new Point([0.38890031504695005, -0.3374825941386163]),
                new Point([0.14401937778315319, -0.8576152837993377])]
        });
        this.expected2dTracePoints.push({
            scale: 0.9,
            projectedUnitVectors: [new Point([0.6844442626793106, 0.3006065902573376]),
                new Point([0.5777730743040868, 0.2948717294336538]),
                new Point([0.42111987483349544, -0.6960147318689115]),
                new Point([0.14273183567375694, -0.5815924983148518])]
        });
        this.expected2dTracePoints.push({
            scale: 1.6,
            projectedUnitVectors: [new Point([0.6346016439294941, 0.4580233794232739]),
                new Point([0.5491971852188988, 0.29473136474521233]),
                new Point([0.5260820120110328, -0.8313867059289775]),
                new Point([0.13748062375493134, -0.1102005069937188])]
        });
        this.expected2dTracePoints.push({
            scale: 2.5,
            projectedUnitVectors: [new Point([0.4531385020572791, 0.6068745698045509]),
                new Point([0.41407991981613557, 0.4962750608664368]),
                new Point([0.7815213409964523, -0.619825398975525]),
                new Point([0.11147964623847292, 0.03508269190624043])]
        });
        this.expected2dTracePoints.push({
            scale: 3.6,
            projectedUnitVectors: [new Point([0.19843355422684578, 0.6982904942112681]),
                new Point([0.19449555991026687, 0.6518258793746413]),
                new Point([0.9587491281250228, -0.28229073994100246]),
                new Point([0.05996424811169943, 0.08846099035857168])]
        });
        this.expected2dTracePoints.push({
            scale: 4.9,
            projectedUnitVectors: [new Point([0.08966054817377281, 0.6724052139934641]),
                new Point([0.09201751446511962, 0.7229344149960059]),
                new Point([0.9912305266235606, -0.13074712890229315]),
                new Point([0.03091611590123808, 0.09023440636804317])]
        });
        this.expected2dTracePoints.push({
            scale: 6.4,
            projectedUnitVectors: [new Point([0.04755062325949049, 0.602849952541657]),
                new Point([0.0500398869133911, 0.7911245501200095]),
                new Point([0.9974602114936664, -0.06977058733276359]),
                new Point([0.01755204923475062, 0.07632788521356752])]
        });
        this.expected2dTracePoints.push({
            scale: 8.1,
            projectedUnitVectors: [new Point([0.028107350062858097, 0.5025279837731735]),
                new Point([0.029994234369487116, 0.8617126930442579]),
                new Point([0.9990968404973528, -0.04062294258926347]),
                new Point([0.01076225279573158, 0.057154498568988146])]
        });
        this.expected2dTracePoints.push({
            scale: 10,
            projectedUnitVectors: [new Point([0.01787163222229783, 0.3893981889434975]),
                new Point([0.019227267245609164, 0.9199093247000198]),
                new Point([0.9996309740763486, -0.024927672616627667]),
                new Point([0.00698803428348653, 0.038916524965570826])]
        });
        this.expected2dTracePoints.push({
            scale: 12.1,
            projectedUnitVectors: [new Point([0.01197698468671412, 0.28771235662246714]),
                new Point([0.012949815900562876, 0.9572501669004975]),
                new Point([0.9998331655075323, -0.015964682576230214]),
                new Point([0.004742916522953394, 0.02527541742397242])]
        });
        this.expected2dTracePoints.push({
            scale: 14.4,
            projectedUnitVectors: [new Point([0.008355493295764217, 0.20946821418770428]),
                new Point([0.009062901894051536, 0.977619431017404]),
                new Point([0.9999184587783715, -0.010665910950064241]),
                new Point([0.003335466283655798, 0.016418090200329184])]
        });
        this.expected2dTracePoints.push({
            scale: 16.9,
            projectedUnitVectors: [new Point([0.006018509438167872, 0.15347405449294968]),
                new Point([0.006541775092897852, 0.9880641926041906]),
                new Point([0.999957573678705, -0.007414139943514283]),
                new Point([0.00241527749408342, 0.010949722362807302])]
        });
        this.expected2dTracePoints.push({
            scale: 19.6,
            projectedUnitVectors: [new Point([0.004450575399762133, 0.11417269383203757]),
                new Point([0.004844446772976933, 0.9934178344970807]),
                new Point([0.9999767551044821, -0.005334368279277283]),
                new Point([0.0017924744868347137, 0.007559537110775172])]
        });
        this.expected2dTracePoints.push({
            scale: 22.5,
            projectedUnitVectors: [new Point([0.0033645922946557204, 0.08643578226831716]),
                new Point([0.0036660263710598636, 0.9962349625618472]),
                new Point([0.9999866970401573, -0.00395043186006185]),
                new Point([0.0013584962472902216, 0.005398981494512551])]
        });
        this.expected2dTracePoints.push({
            scale: 25.6,
            projectedUnitVectors: [new Point([0.0025920913885672378, 0.06656499140761375]),
                new Point([0.0028263462498283967, 0.9977696770051651]),
                new Point([0.9999920967388805, -0.0029967754591995305]),
                new Point([0.0010484697835320609, 0.003974028781945292])]
        });
        this.expected2dTracePoints.push({
            scale: 28.9,
            projectedUnitVectors: [new Point([0.0020299313422870975, 0.05207632066001293]),
                new Point([0.0022145457887927803, 0.9986359017716592]),
                new Point([0.9999951495967623, -0.0023197155876670795]),
                new Point([0.0008221609599688137, 0.003001905863655536])]
        });
        this.expected2dTracePoints.push({
            scale: 32.4,
            projectedUnitVectors: [new Point([0.001612692937266601, 0.041325203337994505]),
                new Point([0.0017600504381628786, 0.9991413901685781]),
                new Point([0.999996936983516, -0.0018267052382927446]),
                new Point([0.0006538100107607596, 0.0023180097167329913])]
        });
        this.expected2dTracePoints.push({
            scale: 36.1,
            projectedUnitVectors: [new Point([0.001297616484658438, 0.03321537044392426]),
                new Point([0.0014166045838070152, 0.9994454872674063]),
                new Point([0.999998016128067, -0.0014598828778433783]),
                new Point([0.0005264625742357337, 0.0018237020242233472])]
        });
    }
}
