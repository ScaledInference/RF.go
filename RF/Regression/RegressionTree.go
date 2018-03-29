//a random forest implemtation in GoLang
package Regression

import (
	"math/rand"
	//"fmt"
	"math"
	"sort"
)

type TreeNode struct{
	ColumnNo int //column number
	Value *float64
	Left *TreeNode
	Right *TreeNode
	Label float64
}

type Tree struct{
	Root *TreeNode
}

func getRandomRange(rs rand.Source, N int, M int) []int{
	tmp := make([]int,N)
	for i:=0;i<N;i++{
		tmp[i]=i
	}
	rr := rand.New(rs)
	for i:=0;i<M;i++{
		j := i + int(rr.Float64()*float64(N-i))
		tmp[i],tmp[j] = tmp[j],tmp[i]
	}

	return tmp[:M]
}

func getSamples(ary [][]float64, index []int)  [][]float64 {
	//fmt.Println("ary",ary)
	result := make([][]float64, len(index))
	for i:=0;i<len(index);i++{
		result[i] = ary[index[i]]
	}
	return result
}


func getLabels(ary []float64, index []int ) []float64{
	result := make([]float64,len(index))
	for i:=0;i<len(index);i++{
		result[i] = ary[index[i]]
	}
	return result
}

func getMSE(labels []float64) float64 {
	if len(labels)==0{
		return 0.0
	}
	total := 0.0
	for _,x := range labels{
		total += x
	}
	avg := total/float64(len(labels))
	mse := 0.0
	for _,x := range labels{
		delta := x - avg
		mse += delta*delta
	}
	mse = mse/float64(len(labels))
	return mse
}


func getBestGain(samples [][]float64, c int, samples_labels []float64,current_mse float64) (float64,float64,int,int){
	var best_value float64
	best_gain := 0.0
	best_total_r := 0
	best_total_l := 0

	uniq_values := make(map[float64]int)
	for i:=0;i<len(samples);i++{
		uniq_values[samples[i][c]] = 1
	}

	uniqueValsList := make([]float64, 0, len(uniq_values))
	for value := range uniq_values {
		uniqueValsList = append(uniqueValsList, value)
	}
	sort.Float64s(uniqueValsList)


	for _, value := range uniqueValsList {
		labels_l := make([]float64,0)
		labels_r := make([]float64,0)
		total_l := 0
		total_r := 0

		for j:=0;j<len(samples);j++{
			if samples[j][c] <=value {
				total_l += 1
				labels_l = append(labels_l,samples_labels[j])
			}else{
				total_r += 1
				labels_r = append(labels_r,samples_labels[j])
			}
		}

		p1 := float64(total_r) / float64(len(samples))
		p2 := float64(total_l) / float64(len(samples))

		new_mse := p1*getMSE(labels_r) + p2*getMSE(labels_l)

		//fmt.Println(new_mse,part_l,part_r)
		mse_gain := current_mse - new_mse

		if mse_gain>=best_gain{
			best_gain = mse_gain
			best_value = value
			best_total_l = total_l
			best_total_r = total_r
		}
	}

	return best_gain, best_value, best_total_l,best_total_r
}

func splitSamples(samples [][]float64, c int, value float64, part_l *[]int, part_r *[]int){
	for j:=0;j<len(samples);j++{
		if samples[j][c] <=value {
			*part_l = append(*part_l,j)
		}else{
			*part_r = append(*part_r,j)
		}
	}
}


func buildTree(rs rand.Source, samples [][]float64, samples_labels []float64, selected_feature_count int) *TreeNode{
	//fmt.Println(len(samples))
	//find a best splitter
	//fmt.Println(samples)
	//fmt.Println("~~~~")
	column_count := len(samples[0])
	//split_count := int(math.Log(float64(column_count)))
	split_count := selected_feature_count
	columns_choosen := getRandomRange(rs, column_count,split_count)

	best_gain := 0.0
	var best_part_l []int = make([]int,0,len(samples))
	var best_part_r []int = make([]int,0,len(samples))
	var best_value *float64
	var best_column int
	var best_total_l int = 0
	var best_total_r int = 0

	current_mse := getMSE(samples_labels)

	for _,c := range columns_choosen{
		//fmt.Println(column_type)
		gain,value,total_l,total_r := getBestGain(samples,c,samples_labels,current_mse)
		//fmt.Println("kkkkk",gain,part_l,part_r)
		if gain>=best_gain{
			best_gain = gain
			best_total_l = total_l
			best_total_r = total_r
			v := value
			best_value = &v
			best_column = c
		}
	}

	if best_gain>0 && best_total_l>0 && best_total_r>0 {
		//fmt.Println(best_part_l,best_part_r)
		node := &TreeNode{}
		node.Value = best_value
		node.ColumnNo = best_column
		splitSamples(samples, best_column, *best_value, &best_part_l,&best_part_r)
		node.Left = buildTree(rs, getSamples(samples,best_part_l),getLabels(samples_labels,best_part_l), selected_feature_count)
		node.Right = buildTree(rs, getSamples(samples,best_part_r),getLabels(samples_labels,best_part_r), selected_feature_count)
		return node
	}

	return genLeafNode(samples_labels)

}

func genLeafNode(labels []float64) *TreeNode{
	total := 0.0
	for _,x := range labels{
		total += x
	}
	avg := total /float64(len(labels))
	node := &TreeNode{}
	node.Label = avg
	//fmt.Println(node)
	return node
}


func predicate(node *TreeNode, input []float64) float64 {
	//fmt.Println("node",node)
	if node.Value == nil{ //leaf node
		return node.Label
	}

	c := node.ColumnNo
	value := input[c]

	if value <= *(node.Value) && node.Left!=nil{
		return predicate(node.Left,input)
	} else if node.Right!=nil{
		return predicate(node.Right,input)
	}
	return math.NaN()
}


func BuildTree(rs rand.Source, inputs [][]float64, labels []float64, samples_count,selected_feature_count int) *Tree{
	samples := make([][]float64,samples_count)
	samples_labels := make([]float64,samples_count)
	rr := rand.New(rs)
	for i:=0;i<samples_count;i++{
		j := int(rr.Float64()*float64(len(inputs)))
		samples[i] = inputs[j]
		samples_labels[i] = labels[j]
	}

	//fmt.Println(samples)
	tree := &Tree{}
	tree.Root = buildTree(rs, samples,samples_labels, selected_feature_count)
	return tree
}



func PredicateTree(tree *Tree, input []float64) float64{
	return predicate(tree.Root,input)
}
